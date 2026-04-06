from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List

from ..models import (
    HiddenTicketState,
    SupportAction,
    SupportObservation,
    SupportPublicState,
    SupportReward,
    SupportState,
    StepResult,
)
from ..task_bank import TASK_ORDER, get_task, next_task_id


class SupportDeskEnvironment:
    """Multi-step support workflow environment with information-revealing actions."""

    def __init__(self) -> None:
        self._state: SupportState | None = None

    @property
    def state(self) -> SupportState:
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state
    @property
    def public_state(self) -> SupportPublicState:
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._build_public_state()
    


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
            revealed_sections=[],
            revealed_data={},
            meaningful_steps=[],
            draft_message=None,
            escalation_state={},
            resolution_state={},
            verification_state={},
            hidden_state=HiddenTicketState(**deepcopy(task.hidden_case)),
            close_attempts=0,
        )
        observation = self._build_observation(task, last_feedback="Fresh case loaded.", done=False)
        self._state.last_observation = observation.model_dump()
        reward = SupportReward(total=0.0, partial={}, penalty=0.0, explanation="Episode reset.")
        return StepResult(
            observation=observation,
            reward=reward,
            done=False,
            info={"task_id": task.task_id, "seed": seed, "done_reason": "reset"},
        )

    def step(self, action: SupportAction) -> StepResult:
        if self._state is None:
            self.reset()
        assert self._state is not None

        if self._state.done:
            raise RuntimeError("Episode already ended. Call /reset to start a new case.")

        task = get_task(self._state.task_id)
        handler = self._dispatch(action.action_type)
        result = handler(task, action)

        self._state.turn += 1
        total = round(sum(result["partial"].values()) - float(result["penalty"]), 4)
        self._state.last_reward = total
        self._state.cumulative_score = self._clamp_score(self._state.cumulative_score + total)
        self._state.best_score = max(self._state.best_score, self._state.cumulative_score)
        self._state.last_action = self._summarize_action(action)
        self._state.violations = list(dict.fromkeys(self._state.violations + list(result.get("violations", []))))
        self._state.transcript.append(
            {
                "turn": self._state.turn,
                "action": self._summarize_action(action),
                "reward": total,
                "partial": result["partial"],
                "penalty": result["penalty"],
                "feedback": result["feedback"],
                "violations": result.get("violations", []),
            }
        )

        done_reason = "continue"
        if result.get("done", False):
            self._state.done = True
            done_reason = "closed"
        elif self._state.turn >= self._state.max_turns:
            self._state.done = True
            done_reason = "max_turns"
            result["feedback"] = f'{result["feedback"]} Turn limit reached before full workflow completion.'.strip()
        else:
            self._state.done = False

        observation = self._build_observation(task, last_feedback=result["feedback"], done=self._state.done)
        self._state.last_observation = observation.model_dump()
        reward = SupportReward(
            total=total,
            partial=result["partial"],
            penalty=float(result["penalty"]),
            explanation=result.get("explanation", result["feedback"]),
        )
        info = {
            "task_id": task.task_id,
            "current_score": self._state.cumulative_score,
            "can_close": self._can_close(task),
            "missing_requirements": self._missing_close_requirements(task),
            "violations": result.get("violations", []),
            "done_reason": done_reason,
        }
        info.update(result.get("info", {}))
        return StepResult(observation=observation, reward=reward, done=self._state.done, info=info)

    def _dispatch(self, action_type: str):
        handlers = {
            "search_customer": self._handle_search_customer,
            "view_order": self._handle_view_order,
            "check_policy": self._handle_check_policy,
            "inspect_previous_tickets": self._handle_previous_tickets,
            "draft_response": self._handle_draft_response,
            "escalate_case": self._handle_escalate_case,
            "take_resolution_action": self._handle_resolution_action,
            "close_ticket": self._handle_close_ticket,
        }
        try:
            return handlers[action_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported action_type: {action_type}") from exc

    def _handle_search_customer(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        if "customer" in self._state.revealed_sections:
            return self._result(
                feedback="Customer record already inspected; use another action to make progress.",
                partial={},
                penalty=0.02,
            )

        customer_view = deepcopy(task.customer_record)
        self._reveal_section("customer", customer_view)
        self._record_meaningful_step("search_customer")
        self._state.verification_state["customer_verified"] = True
        return self._result(
            feedback="Customer account details revealed.",
            partial={"info_reveal": 0.10, "workflow_progress": 0.04},
            info={"revealed_section": "customer"},
        )

    def _handle_view_order(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        if task.order_record is None and task.payment_record is None:
            return self._result(
                feedback="No linked order or payment record exists for this case.",
                partial={},
                penalty=0.03,
            )

        already_had_order = "order" in self._state.revealed_sections
        already_had_payment = "payment" in self._state.revealed_sections
        if already_had_order and (task.payment_record is None or already_had_payment):
            return self._result(
                feedback="Order and payment records were already inspected.",
                partial={},
                penalty=0.02,
            )

        if task.order_record is not None:
            self._reveal_section("order", deepcopy(task.order_record))
            self._state.verification_state["order_verified"] = True
        if task.payment_record is not None:
            self._reveal_section("payment", deepcopy(task.payment_record))
            self._state.verification_state["payment_verified"] = True

        self._record_meaningful_step("view_order")
        return self._result(
            feedback="Order and payment details revealed for verification.",
            partial={"info_reveal": 0.12, "verification": 0.05},
            info={"revealed_sections": [s for s in ["order", "payment"] if s in self._state.revealed_sections]},
        )

    def _handle_check_policy(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        policy_docs = deepcopy(task.policy_docs)
        if action.policy_key:
            selected = {k: v for k, v in policy_docs.items() if k == action.policy_key}
            if not selected:
                return self._result(
                    feedback=f"No policy matched key '{action.policy_key}'.",
                    partial={},
                    penalty=0.03,
                )
            policy_docs = selected

        existing = self._state.revealed_data.get("policy", {}) if "policy" in self._state.revealed_sections else {}
        merged = dict(existing)
        new_keys = []
        for key, value in policy_docs.items():
            if key not in merged:
                new_keys.append(key)
            merged[key] = value
        if not new_keys and "policy" in self._state.revealed_sections:
            return self._result(
                feedback="Those policy documents were already inspected.",
                partial={},
                penalty=0.02,
            )

        self._reveal_section("policy", merged)
        self._record_meaningful_step("check_policy")
        self._state.verification_state["policy_checked"] = True
        return self._result(
            feedback="Relevant policy guidance revealed.",
            partial={"policy_lookup": 0.12, "workflow_progress": 0.04},
            info={"policy_keys": list(policy_docs.keys())},
        )

    def _handle_previous_tickets(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        if "previous_tickets" in self._state.revealed_sections:
            return self._result(
                feedback="Previous tickets are already visible.",
                partial={},
                penalty=0.02,
            )
        self._reveal_section("previous_tickets", deepcopy(task.previous_tickets))
        self._record_meaningful_step("inspect_previous_tickets")
        return self._result(
            feedback="Previous support interactions revealed.",
            partial={"history_lookup": 0.08, "workflow_progress": 0.03},
            info={"revealed_section": "previous_tickets"},
        )

    def _handle_draft_response(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        message = (action.message or "").strip()
        if not message:
            return self._result(
                feedback="draft_response requires a non-empty customer-facing message.",
                partial={},
                penalty=0.06,
            )

        lower_message = message.lower()
        violations = [phrase for phrase in task.forbidden_phrases if phrase.lower() in lower_message]
        required_hits = self._draft_hits(task, lower_message)
        required_total = len(task.expected_resolution.get("draft_must_include", []))

        self._state.draft_message = message
        self._record_meaningful_step("draft_response")
        self._state.verification_state["draft_created"] = True
        self._state.verification_state["draft_hits"] = required_hits
        self._state.verification_state["draft_total"] = required_total

        partial = {"draft_quality": 0.08}
        if required_total > 0:
            partial["draft_completeness"] = round(0.06 * (required_hits / required_total), 4)

        penalty = 0.0
        feedback = "Customer-facing draft saved."
        if violations:
            penalty += min(0.18, 0.06 * len(violations))
            self._state.violations = list(dict.fromkeys(self._state.violations + violations))
            feedback = "Draft saved, but it includes policy-risky language that should be corrected."
        elif required_total > 0 and required_hits < required_total:
            feedback = f"Draft saved, but it is still missing {required_total - required_hits} expected detail(s)."
        else:
            feedback = "Customer-facing draft saved and looks workflow-complete."

        return self._result(
            feedback=feedback,
            partial=partial,
            violations=violations,
            info={"draft_hits": required_hits, "draft_total": required_total},
        )
    def _handle_escalate_case(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        hidden = self._state.hidden_state

        payload = {
            "team": action.team,
            "priority": action.priority,
            "severity": action.severity,
            "status": action.status,
            "tags": list(action.tags),
            "internal_note": action.internal_note,
        }
        self._state.escalation_state = payload
        self._record_meaningful_step("escalate_case")
        self._state.verification_state["escalated"] = True

        partial = {"escalation": 0.08}
        penalty = 0.0
        feedback = "Case escalation recorded."

        if hidden.escalation_required:
            if action.team == hidden.correct_team:
                partial["team_match"] = 0.05
            else:
                penalty += 0.04

            if action.priority == hidden.correct_priority:
                partial["priority_match"] = 0.04
            else:
                penalty += 0.03

            if hidden.correct_severity:
                if action.severity == hidden.correct_severity:
                    partial["severity_match"] = 0.05
                else:
                    penalty += 0.05

            if penalty > 0:
                feedback = "Escalation recorded, but the escalation details may still need correction."
            else:
                feedback = "Escalation recorded with incident-appropriate routing."
        else:
            penalty += 0.04
            feedback = "Escalation recorded, but this case may not require escalation."

        return self._result(
            feedback=feedback,
            partial=partial,
            penalty=penalty,
        )
    def _handle_resolution_action(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        hidden = self._state.hidden_state

        if not action.resolution_type:
            return self._result(
                feedback="take_resolution_action requires resolution_type.",
                partial={},
                penalty=0.06,
            )

        self._state.resolution_state = {
            "resolution_type": action.resolution_type,
            "resolution_payload": deepcopy(action.resolution_payload),
            "status": action.status,
            "tags": list(action.tags),
            "internal_note": action.internal_note,
        }
        self._record_meaningful_step("take_resolution_action")
        self._state.verification_state["resolution_taken"] = True

        partial = {"resolution_action": 0.10}
        penalty = 0.0
        feedback = "Resolution action recorded."

        hidden.issue_resolved = False
        hidden.wrong_action_caused_damage = False

        if action.resolution_type == hidden.correct_resolution_type:
            partial["resolution_match"] = 0.10
            hidden.issue_resolved = True
        else:
            penalty += 0.08
            feedback = "Resolution action recorded, but it may not correctly address the case."

        expected_tags = set(task.expected_resolution.get("required_tags", []))
        provided_tags = set(action.tags)
        if expected_tags:
            matched = len(expected_tags & provided_tags)
            partial["tag_alignment"] = round(0.04 * (matched / len(expected_tags)), 4)

        if action.resolution_type == "issue_refund":
            if hidden.correct_resolution_type != "issue_refund":
                hidden.wrong_action_caused_damage = True
                hidden.issue_resolved = False
                penalty += 0.08
                feedback = "Resolution action recorded, but it conflicts with the case needs."
            elif hidden.refund_eligible is False:
                hidden.wrong_action_caused_damage = True
                hidden.issue_resolved = False
                penalty += 0.08
                feedback = "Resolution action recorded, but it conflicts with the case policy."
            elif hidden.refund_eligible is True:
                partial["policy_safe_resolution"] = 0.04

        if hidden.correct_resolution_type == "publish_status_update":
            workaround_required = bool(task.expected_resolution.get("workaround_available"))
            workaround_text = str(action.resolution_payload.get("workaround", "")).strip()

            if action.resolution_type == "publish_status_update":
                if workaround_required and workaround_text:
                    partial["workaround_recorded"] = 0.04
                    hidden.issue_resolved = True
                elif workaround_required and not workaround_text:
                    penalty += 0.03
                    hidden.issue_resolved = False
                    feedback = "Status action recorded, but an important operational detail is still missing."

        return self._result(
            feedback=feedback,
            partial=partial,
            penalty=penalty,
        )
    
    def _handle_close_ticket(self, task, action: SupportAction) -> Dict[str, Any]:
        assert self._state is not None
        hidden = self._state.hidden_state

        self._state.close_attempts += 1
        missing = self._missing_close_requirements(task)
        if missing:
            return self._result(
                feedback="Cannot close yet: " + "; ".join(missing) + ".",
                partial={},
                penalty=0.12,
                info={"missing_requirements": missing},
                done=False,
            )

        self._record_meaningful_step("close_ticket")
        expected_status = task.expected_resolution.get("status")
        if expected_status:
            self._state.resolution_state["final_status"] = expected_status

        return self._result(
            feedback=hidden.close_note or "Ticket closed after successful workflow completion.",
            partial={"close_success": 0.24},
            penalty=0.0,
            done=True,
        )

    def _build_public_state(self) -> SupportPublicState:
        assert self._state is not None
        return SupportPublicState(
            episode_id=self._state.episode_id,
            task_id=self._state.task_id,
            turn=self._state.turn,
            max_turns=self._state.max_turns,
            done=self._state.done,
            cumulative_score=self._state.cumulative_score,
            best_score=self._state.best_score,
            last_reward=self._state.last_reward,
            revealed_sections=deepcopy(self._state.revealed_sections),
            revealed_data=deepcopy(self._state.revealed_data),
            close_attempts=self._state.close_attempts,
            draft_exists=bool(self._state.draft_message),
            escalation_recorded=bool(self._state.escalation_state),
            resolution_recorded=bool(self._state.resolution_state),
            action_history=self._public_action_history(),
            can_close=self._can_close(get_task(self._state.task_id)),
        )

    
    def _build_observation(self, task, last_feedback: str, done: bool = False) -> SupportObservation:
        assert self._state is not None
        return SupportObservation(
            task_id=task.task_id,
            title=task.title,
            difficulty=task.difficulty,
            turn=self._state.turn,
            remaining_turns=max(0, self._state.max_turns - self._state.turn),
            inbox_summary=task.public_inbox_summary,
            constraints=deepcopy(task.constraints),
            open_questions=self._current_open_questions(task),
            available_actions=list(task.allowed_actions),
            revealed_data=deepcopy(self._state.revealed_data),
            action_history=self._public_action_history(),
            last_feedback=last_feedback,
            current_score=self._state.cumulative_score,
            done=done,
            can_close=self._can_close(task),
        )

    def _current_open_questions(self, task) -> List[str]:
        assert self._state is not None
        questions: List[str] = []
        if "customer" not in self._state.revealed_sections:
            questions.append("Customer identity and account context still need verification.")
        if (task.order_record is not None or task.payment_record is not None) and (
            "order" not in self._state.revealed_sections or (task.payment_record is not None and "payment" not in self._state.revealed_sections)
        ):
            questions.append("Order or payment evidence still needs inspection.")
        if "policy" not in self._state.revealed_sections:
            questions.append("Relevant policy guidance has not been checked yet.")
        if "inspect_previous_tickets" in task.required_actions and "previous_tickets" not in self._state.revealed_sections:
            questions.append("Prior related tickets may still contain useful context.")
        if not self._state.resolution_state:
            questions.append("A concrete operational resolution has not been recorded yet.")
        if not self._state.draft_message:
            questions.append("A customer-facing draft response is still needed.")
        if "escalate_case" in task.required_actions and not self._state.escalation_state:
            questions.append("Required escalation details are still missing.")
        return questions or ["All critical facts appear covered; you can consider closing the ticket."]

    def _public_action_history(self) -> List[Dict[str, Any]]:
        assert self._state is not None
        history: List[Dict[str, Any]] = []
        for entry in self._state.transcript[-6:]:
            history.append(
                {
                    "turn": entry["turn"],
                    "action_type": entry["action"].get("action_type"),
                    "reward": entry["reward"],
                    "feedback": entry["feedback"],
                }
            )
        return history

    def _can_close(self, task) -> bool:
        return len(self._missing_close_requirements(task)) == 0
    
    def _missing_close_requirements(self, task) -> List[str]:
        assert self._state is not None
        hidden = self._state.hidden_state
        missing: List[str] = []

        if len(self._state.meaningful_steps) < task.min_meaningful_steps:
            missing.append(
                f"need at least {task.min_meaningful_steps} meaningful steps (have {len(self._state.meaningful_steps)})"
            )

        taken = {entry["action"].get("action_type") for entry in self._state.transcript}
        for action_name in task.required_actions:
            if action_name == "close_ticket":
                continue
            if action_name not in taken:
                missing.append(f"required workflow step '{action_name}' not completed")

        revealed = set(self._state.revealed_sections)
        for section in task.required_revealed_sections:
            if section not in revealed:
                missing.append(f"required data section '{section}' not revealed")

        if not self._state.resolution_state:
            missing.append("a concrete operational resolution has not been recorded")
        elif not hidden.issue_resolved:
            missing.append("the recorded resolution does not yet appear workflow-complete")

        if hidden.escalation_required:
            if not self._state.escalation_state:
                missing.append("required escalation details are still missing")
            else:
                escalation_ok = True
                if hidden.correct_team and self._state.escalation_state.get("team") != hidden.correct_team:
                    escalation_ok = False
                if hidden.correct_priority and self._state.escalation_state.get("priority") != hidden.correct_priority:
                    escalation_ok = False
                if hidden.correct_severity and self._state.escalation_state.get("severity") != hidden.correct_severity:
                    escalation_ok = False
                if not escalation_ok:
                    missing.append("escalation details are incomplete or incorrect")

        if not self._state.draft_message:
            missing.append("draft response missing")
        elif not self._draft_requirements_met(task):
            missing.append("draft response is missing required customer-facing details")

        if hidden.wrong_action_caused_damage:
            missing.append("a previous action introduced case risk and should be corrected before closing")

        return missing
    def _draft_requirements_met(self, task) -> bool:
        assert self._state is not None
        message = (self._state.draft_message or "").lower()
        if not message:
            return False
        if any(phrase.lower() in message for phrase in task.forbidden_phrases):
            return False
        required = task.expected_resolution.get("draft_must_include", [])
        return self._draft_hits(task, message) >= len(required)

    def _draft_hits(self, task, lower_message: str) -> int:
        hits = 0
        for phrase in task.expected_resolution.get("draft_must_include", []):
            normalized = phrase.lower()
            if normalized == "no eta":
                if (
                    "no eta" in lower_message
                    or "do not have an eta" in lower_message
                    or "don't have an eta" in lower_message
                    or "dont have an eta" in lower_message
                ):
                    hits += 1
            elif normalized == "workaround":
                if "workaround" in lower_message or "failover endpoint" in lower_message:
                    hits += 1
            elif normalized in lower_message:
                hits += 1
        return hits

    def _reveal_section(self, section: str, payload: Any) -> None:
        assert self._state is not None
        self._state.revealed_data[section] = payload
        if section not in self._state.revealed_sections:
            self._state.revealed_sections.append(section)

    def _record_meaningful_step(self, marker: str) -> None:
        assert self._state is not None
        if marker not in self._state.meaningful_steps:
            self._state.meaningful_steps.append(marker)

    def _summarize_action(self, action: SupportAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _result(
    self,
    *,
    feedback: str,
    partial: Dict[str, float],
    penalty: float = 0.0,
    info: Dict[str, Any] | None = None,
    violations: List[str] | None = None,
    done: bool = False,
    explanation: str | None = None,
) -> Dict[str, Any]:
        return {
            "feedback": feedback,
            "partial": partial,
            "penalty": penalty,
            "info": info or {},
            "violations": violations or [],
            "done": done,
            "explanation": explanation or feedback,
        }

    @staticmethod
    def _clamp_score(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return round(value, 4)


class SupportDeskAppState:
    """Container-friendly singleton wrapper used by the FastAPI app."""

    def __init__(self) -> None:
        self.env = SupportDeskEnvironment()

    def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "task_count": len(TASK_ORDER)}