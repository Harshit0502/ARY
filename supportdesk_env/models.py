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
    ] = Field(...)

    query: Optional[str] = None
    customer_id: Optional[str] = None
    order_id: Optional[str] = None
    policy_key: Optional[str] = None

    team: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    severity: Optional[Literal["sev4", "sev3", "sev2", "sev1"]] = None
    status: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    message: Optional[str] = None
    internal_note: Optional[str] = None

    resolution_type: Optional[str] = None
    resolution_payload: Dict[str, Any] = Field(default_factory=dict)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SupportObservation(OpenEnvObservation):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    turn: int = Field(..., ge=0)
    remaining_turns: int = Field(..., ge=0)

    inbox_summary: str
    constraints: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

    available_actions: List[str] = Field(default_factory=list)
    revealed_data: Dict[str, Any] = Field(default_factory=dict)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)

    last_feedback: str = ""
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = False
    can_close: bool = False


class SupportReward(OpenEnvReward):
    model_config = ConfigDict(extra="forbid")

    total: float
    partial: Dict[str, float] = Field(default_factory=dict)
    penalty: float = 0.0
    explanation: str = ""


class HiddenTicketState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actual_issue_type: str = "unknown"
    required_policy_keys: List[str] = Field(default_factory=list)
    refund_eligible: Optional[bool] = None
    escalation_required: bool = False
    customer_sentiment: Literal["unknown", "calm", "frustrated", "angry", "urgent"] = "unknown"
    issue_resolved: bool = False
    wrong_action_caused_damage: bool = False

    correct_resolution_type: Optional[str] = None
    correct_team: Optional[str] = None
    correct_priority: Optional[str] = None
    correct_severity: Optional[str] = None
    close_note: Optional[str] = None


class SupportPublicState(OpenEnvState):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str

    turn: int = Field(default=0, ge=0)
    max_turns: int = Field(default=6, ge=1)
    done: bool = False

    cumulative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_reward: float = 0.0

    revealed_sections: List[str] = Field(default_factory=list)
    revealed_data: Dict[str, Any] = Field(default_factory=dict)

    close_attempts: int = Field(default=0, ge=0)
    draft_exists: bool = False
    escalation_recorded: bool = False
    resolution_recorded: bool = False

    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    can_close: bool = False


class SupportState(OpenEnvState):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str

    turn: int = Field(default=0, ge=0)
    max_turns: int = Field(default=6, ge=1)
    done: bool = False

    cumulative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_reward: float = 0.0

    transcript: List[Dict[str, Any]] = Field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None
    last_observation: Optional[Dict[str, Any]] = None
    violations: List[str] = Field(default_factory=list)

    revealed_sections: List[str] = Field(default_factory=list)
    revealed_data: Dict[str, Any] = Field(default_factory=dict)
    meaningful_steps: List[str] = Field(default_factory=list)

    draft_message: Optional[str] = None
    escalation_state: Dict[str, Any] = Field(default_factory=dict)
    resolution_state: Dict[str, Any] = Field(default_factory=dict)
    verification_state: Dict[str, Any] = Field(default_factory=dict)

    hidden_state: HiddenTicketState = Field(default_factory=HiddenTicketState)

    close_attempts: int = Field(default=0, ge=0)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: SupportObservation
    reward: SupportReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)