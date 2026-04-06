from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:  # Optional compatibility with OpenEnv core if present.
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State as OpenEnvState
    from openenv.core.env_server.types import Reward as OpenEnvReward
except Exception:  # pragma: no cover - fallback for standalone execution
    OpenEnvAction = BaseModel
    OpenEnvObservation = BaseModel
    OpenEnvState = BaseModel
    OpenEnvReward = BaseModel


class SupportAction(OpenEnvAction):
    """Agent action used to work a support ticket."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["classify", "draft_reply", "escalate", "resolve", "finalize"] = Field(
        default="classify",
        description="High-level intent for this step.",
    )
    issue_type: Optional[str] = Field(default=None, description="Predicted issue category.")
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = Field(
        default=None,
        description="Chosen priority.",
    )
    team: Optional[str] = Field(default=None, description="Owning team to route the ticket to.")
    severity: Optional[Literal["sev4", "sev3", "sev2", "sev1"]] = Field(
        default=None,
        description="Incident severity, if applicable.",
    )
    status: Optional[str] = Field(default=None, description="Ticket or incident status update.")
    tags: List[str] = Field(default_factory=list, description="Tags to apply to the case.")
    message: Optional[str] = Field(default=None, description="Customer-facing or internal response draft.")
    internal_note: Optional[str] = Field(default=None, description="Internal reasoning or handoff note.")
    refund_amount: Optional[float] = Field(default=None, ge=0.0, description="Refund amount to issue.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Self-reported confidence.")


class SupportObservation(OpenEnvObservation):
    """Observation returned to the agent after reset/step."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., description="Current hidden task identifier.")
    title: str = Field(..., description="Human-readable task title.")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Task difficulty.")
    turn: int = Field(..., ge=0, description="Current turn number.")
    remaining_turns: int = Field(..., ge=0, description="Turns left before automatic termination.")
    inbox_summary: str = Field(..., description="User-facing summary of the ticket or incident.")
    open_questions: List[str] = Field(default_factory=list, description="Important unanswered questions.")
    constraints: List[str] = Field(default_factory=list, description="Policy and operational constraints.")
    last_feedback: str = Field(default="", description="Feedback from the most recent action.")
    current_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Best score so far.")
    done: bool = Field(default=False, description="Whether the episode has ended.")
    suggested_fields: List[str] = Field(
        default_factory=list,
        description="Fields the agent should consider filling next.",
    )


class SupportReward(OpenEnvReward):
    """Reward payload returned by the environment."""

    model_config = ConfigDict(extra="forbid")

    total: float = Field(..., description="Total scalar reward for the last step.")
    partial: Dict[str, float] = Field(default_factory=dict, description="Per-criterion contribution.")
    penalty: float = Field(default=0.0, description="Penalty applied for undesirable behavior.")
    explanation: str = Field(default="", description="Short human-readable summary.")


class SupportState(OpenEnvState):
    """Internal environment state."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(..., description="Episode identifier.")
    task_id: str = Field(..., description="Current task identifier.")
    turn: int = Field(default=0, ge=0)
    max_turns: int = Field(default=4, ge=1)
    done: bool = Field(default=False)
    cumulative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_reward: float = Field(default=0.0)
    transcript: List[Dict[str, Any]] = Field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = Field(default=None)
    last_observation: Optional[Dict[str, Any]] = Field(default=None)
    violations: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    """Standard step result returned by the API."""

    model_config = ConfigDict(extra="forbid")

    observation: SupportObservation
    reward: SupportReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
