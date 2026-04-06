from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, Optional

from pydantic import ValidationError

from ..grader import build_feedback, score_action, summarize_action
from ..models import SupportAction, SupportObservation, SupportReward, SupportState, StepResult
from ..task_bank import TASK_ORDER, get_task, next_task_id


class SupportDeskEnvironment:
    """A support operations simulation built for OpenEnv-style agents.

    The environment cycles through three real-world tasks:
    login triage, billing refund handling, and outage incident coordination.
    """

    def __init__(self) -> None:
        self._state: SupportState | None = None

    @property
    def state(self) -> SupportState:
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state

    def reset(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> StepResult:
        if task_id is None:
            if self._state is None:
                task_id = TASK_ORDER[0]
            else:
                task_id = next_task_id(self._state.task_id)

        task = get_task(task_id)
        episode_id = episode_id or str(uuid.uuid4())
        self._state = SupportState(
            episode_id=episode_id,
            task_id=task.task_id,
            turn=0,
            max_turns=task.max_turns,
            done=False,
            cumulative_score=0.0,
            best_score=0.0,
            last_reward=0.0,
            transcript=[],
            last_action=None,
            last_observation=None,
            violations=[],
        )
        observation = self._build_observation(task, last_feedback="Fresh case loaded.")
        self._state.last_observation = observation.model_dump()
        reward = SupportReward(total=0.0, partial={}, penalty=0.0, explanation="Episode reset.")
        return StepResult(
            observation=observation,
            reward=reward,
            done=False,
            info={"task_id": task.task_id, "seed": seed},
        )

    def step(self, action: SupportAction) -> StepResult:
        if self._state is None:
            self.reset()
        assert self._state is not None
        task = get_task(self._state.task_id)

        rubric = score_action(task, action)
        current_score = float(rubric["score"])
        previous_best = self._state.best_score
        reward_value = current_score - previous_best

        # Small time pressure to discourage endless looping.
        if self._state.turn >= task.max_turns - 1:
            reward_value -= 0.03
        if rubric["violations"]:
            reward_value -= min(0.15, 0.03 * len(rubric["violations"]))

        self._state.turn += 1
        self._state.best_score = max(previous_best, current_score)
        self._state.cumulative_score = self._state.best_score
        self._state.last_reward = reward_value
        self._state.last_action = summarize_action(action)
        self._state.transcript.append(
            {
                "turn": self._state.turn,
                "action": summarize_action(action),
                "score": current_score,
                "reward": reward_value,
                "violations": rubric["violations"],
            }
        )
        self._state.violations = list(dict.fromkeys(self._state.violations + list(rubric["violations"])))

        done = self._state.turn >= self._state.max_turns or current_score >= 0.98
        self._state.done = done

        feedback = build_feedback(task, rubric)
        observation = self._build_observation(task, last_feedback=feedback, done=done)
        self._state.last_observation = observation.model_dump()
        reward = SupportReward(
            total=reward_value,
            partial=rubric["criteria"],
            penalty=float(rubric["penalty"]),
            explanation=rubric["explanation"],
        )
        info = {
            "task_id": task.task_id,
            "current_score": current_score,
            "previous_best": previous_best,
            "violations": rubric["violations"],
            "rubric": rubric["criteria"],
            "feedback": feedback,
            "done_reason": "max_turns" if self._state.turn >= self._state.max_turns else ("threshold" if current_score >= 0.98 else "continue"),
        }
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def _build_observation(self, task, last_feedback: str, done: bool = False) -> SupportObservation:
        assert self._state is not None
        return SupportObservation(
            task_id=task.task_id,
            title=task.title,
            difficulty=task.difficulty,
            turn=self._state.turn,
            remaining_turns=max(0, self._state.max_turns - self._state.turn),
            inbox_summary=task.inbox_summary,
            open_questions=deepcopy(task.open_questions),
            constraints=deepcopy(task.constraints),
            last_feedback=last_feedback,
            current_score=self._state.best_score,
            done=done,
            suggested_fields=["action_type", "issue_type", "priority", "team", "severity", "status", "tags", "message", "internal_note"],
        )


class SupportDeskAppState:
    """Container-friendly singleton wrapper used by the FastAPI app."""

    def __init__(self) -> None:
        self.env = SupportDeskEnvironment()

    def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "task_count": len(TASK_ORDER)}
