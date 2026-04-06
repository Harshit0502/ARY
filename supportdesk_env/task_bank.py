from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: str
    inbox_summary: str
    open_questions: List[str]
    constraints: List[str]
    target_team: str
    target_priority: str
    target_issue_type: str
    target_severity: str | None
    target_status: str
    required_phrases: List[str]
    required_tags: List[str]
    forbidden_phrases: List[str]
    max_turns: int = 4


TASKS: Dict[str, TaskSpec] = {
    "login_lockout": TaskSpec(
        task_id="login_lockout",
        title="Password-reset login lockout triage",
        difficulty="easy",
        inbox_summary=(
            "A customer cannot log in after a password reset. The account is not locked, but they report the "
            "reset email may have landed in spam and they are unsure whether 2FA is also blocking access."
        ),
        open_questions=[
            "Which support queue should own the case?",
            "What immediate next step should be sent to the customer?",
        ],
        constraints=[
            "Do not ask for the customer's password.",
            "Do not suggest turning off 2FA globally.",
            "Use a calm, supportive tone.",
        ],
        target_team="account_access",
        target_priority="medium",
        target_issue_type="login_lockout",
        target_severity=None,
        target_status="investigating",
        required_phrases=["reset link", "check spam", "2fa"],
        required_tags=["login", "password-reset"],
        forbidden_phrases=["password", "disable 2fa", "we cannot help"],
        max_turns=4,
    ),
    "duplicate_charge_refund": TaskSpec(
        task_id="duplicate_charge_refund",
        title="Duplicate billing charge resolution",
        difficulty="medium",
        inbox_summary=(
            "A customer was charged twice for the same monthly subscription. The second charge happened two days ago "
            "and the customer already attached a screenshot of the card statement. They want the duplicate charge reversed."
        ),
        open_questions=[
            "Which team should own the case?",
            "What should the customer be told about the refund timeline?",
            "Should the billing record be marked for duplicate-charge review?",
        ],
        constraints=[
            "Do not ask the customer to resend payment details.",
            "Refunds must go to the original payment method.",
            "Set expectations that bank processing can take a few business days.",
        ],
        target_team="billing",
        target_priority="high",
        target_issue_type="duplicate_charge",
        target_severity=None,
        target_status="pending_refund",
        required_phrases=["refund", "original payment method", "3-5 business days"],
        required_tags=["billing", "refund", "duplicate-charge"],
        forbidden_phrases=["immediate refund", "send card number", "wire transfer"],
        max_turns=4,
    ),
    "eu_outage_incident": TaskSpec(
        task_id="eu_outage_incident",
        title="EU region outage incident coordination",
        difficulty="hard",
        inbox_summary=(
            "Monitoring shows a service outage affecting about 20% of European users. The incident is ongoing, the "
            "root cause is not confirmed, and the status page has not yet been updated. A customer-facing update is required."
        ),
        open_questions=[
            "What severity should be assigned?",
            "Which team should own the incident bridge?",
            "What should the public update say about ETA and workaround?",
        ],
        constraints=[
            "Do not claim the incident is fixed before verification.",
            "Do not promise an exact ETA.",
            "The public update should mention a workaround only if one is available.",
            "Use incident-bridge language appropriate for an active outage.",
        ],
        target_team="incident_management",
        target_priority="urgent",
        target_issue_type="regional_outage",
        target_severity="sev2",
        target_status="investigating",
        required_phrases=["status page", "no eta", "workaround"],
        required_tags=["incident", "outage", "eu"],
        forbidden_phrases=["fixed", "resolved", "exact eta", "100% restored"],
        max_turns=4,
    ),
}

TASK_ORDER = ["login_lockout", "duplicate_charge_refund", "eu_outage_incident"]


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown task_id: {task_id!r}") from exc


def next_task_id(current_task_id: str | None) -> str:
    if current_task_id is None:
        return TASK_ORDER[0]
    try:
        idx = TASK_ORDER.index(current_task_id)
    except ValueError:
        return TASK_ORDER[0]
    return TASK_ORDER[(idx + 1) % len(TASK_ORDER)]


def public_task_manifest(task: TaskSpec) -> dict:
    return {
        "task_id": task.task_id,
        "title": task.title,
        "difficulty": task.difficulty,
        "inbox_summary": task.inbox_summary,
        "open_questions": task.open_questions,
        "constraints": task.constraints,
        "max_turns": task.max_turns,
    }

def public_tasks() -> dict:
    return {task_id: public_task_manifest(task) for task_id, task in TASKS.items()}