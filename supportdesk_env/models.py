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
    """A single operational action in the support workflow."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "search_customer",
        "view_order",
        "check_policy",
        "inspect_previous_tickets",
        "draft_response",
        "escalate_case",
        "take_resolution_action",
        "close_ticket",
    ] = Field(
        ...,
        description="Exactly one next step the agent wants to perform.",
    )

    query: Optional[str] = Field(
        default=None,
        description="Free-text lookup query, mainly for search_customer.",
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier if already known.",
    )
    order_id: Optional[str] = Field(
        default=None,
        description="Order identifier if already known.",
    )
    policy_key: Optional[str] = Field(
        default=None,
        description="Policy document or policy topic to inspect.",
    )

    team: Optional[str] = Field(
        default=None,
        description="Owning team for escalation or routing.",
    )
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = Field(
        default=None,
        description="Priority selected for escalation or follow-up.",
    )
    severity: Optional[Literal["sev4", "sev3", "sev2", "sev1"]] = Field(
        default=None,
        description="Incident severity for outage or incident cases.",
    )
    status: Optional[str] = Field(
        default=None,
        description="Status update for escalation or resolution tracking.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags to attach to the case.",
    )

    message: Optional[str] = Field(
        default=None,
        description="Customer-facing draft for draft_response.",
    )
    internal_note: Optional[str] = Field(
        default=None,
        description="Internal note for escalation, verification, or audit trail.",
    )

    resolution_type: Optional[str] = Field(
        default=None,
        description="Operational resolution, e.g. issue_refund, reissue_reset_link, workaround_shared.",
    )
    resolution_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured inputs for the chosen resolution action.",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence for the selected next step.",
    )


class SupportObservation(OpenEnvObservation):
    """Observation exposed to the agent after each step."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., description="Current task identifier.")
    title: str = Field(..., description="Human-readable task title.")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Task difficulty.")
    turn: int = Field(..., ge=0, description="Current turn number.")
    remaining_turns: int = Field(..., ge=0, description="Turns left before automatic termination.")

    inbox_summary: str = Field(
        ...,
        description="Minimal user-visible summary available at reset.",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Operational or policy constraints the agent should obey.",
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="What remains unknown and may need inspection.",
    )

    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types currently allowed by the environment.",
    )
    revealed_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Facts the agent has uncovered so far.",
    )
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compact trace of prior actions and outcomes.",
    )

    last_feedback: str = Field(
        default="",
        description="Feedback from the previous step.",
    )
    current_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current progress score, not final correctness only.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    can_close: bool = Field(
        default=False,
        description="Whether the ticket currently satisfies close prerequisites.",
    )


class SupportReward(OpenEnvReward):
    """Reward payload returned by the environment."""

    model_config = ConfigDict(extra="forbid")

    total: float = Field(..., description="Total scalar reward for the last step.")
    partial: Dict[str, float] = Field(default_factory=dict, description="Per-criterion contribution.")
    penalty: float = Field(default=0.0, description="Penalty applied for undesirable behavior.")
    explanation: str = Field(default="", description="Short human-readable summary.")


class SupportState(OpenEnvState):
    """Internal environment state for a multi-step support workflow."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(..., description="Episode identifier.")
    task_id: str = Field(..., description="Current task identifier.")

    turn: int = Field(default=0, ge=0)
    max_turns: int = Field(default=6, ge=1)
    done: bool = Field(default=False)

    cumulative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_reward: float = Field(default=0.0)

    transcript: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full internal action transcript.",
    )
    last_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Last action summary.",
    )
    last_observation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Last observation snapshot.",
    )
    violations: List[str] = Field(
        default_factory=list,
        description="Accumulated policy or workflow violations.",
    )

    revealed_sections: List[str] = Field(
        default_factory=list,
        description="Named data sections already unlocked, e.g. customer, order, policy.",
    )
    revealed_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="All information uncovered so far by the agent.",
    )
    meaningful_steps: List[str] = Field(
        default_factory=list,
        description="Unique meaningful step markers used for close gating.",
    )

    draft_message: Optional[str] = Field(
        default=None,
        description="Latest customer-facing response draft.",
    )
    escalation_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tracks escalation details and whether escalation happened.",
    )
    resolution_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tracks applied operational resolution.",
    )
    verification_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tracks whether key facts have been verified.",
    )

    close_attempts: int = Field(
        default=0,
        ge=0,
        description="How many times the agent tried to close the ticket.",
    )


class StepResult(BaseModel):
    """Standard step result returned by the API."""

    model_config = ConfigDict(extra="forbid")

    observation: SupportObservation
    reward: SupportReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)