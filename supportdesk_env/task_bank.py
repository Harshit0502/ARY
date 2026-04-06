from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_ALLOWED_ACTIONS: List[str] = [
    "search_customer",
    "view_order",
    "check_policy",
    "inspect_previous_tickets",
    "draft_response",
    "escalate_case",
    "take_resolution_action",
    "close_ticket",
]

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: str

    public_inbox_summary: str
    public_open_questions: List[str]
    constraints: List[str]

    customer_record: Dict[str, Any]
    order_record: Optional[Dict[str, Any]]
    payment_record: Optional[Dict[str, Any]]
    previous_tickets: List[Dict[str, Any]]
    policy_docs: Dict[str, str]

    expected_resolution: Dict[str, Any]
    required_actions: List[str]
    required_revealed_sections: List[str]
    min_meaningful_steps: int
    forbidden_phrases: List[str]

    max_turns: int = 6
    allowed_actions: List[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ACTIONS))
    hidden_case: Dict[str, Any] = field(default_factory=dict)
    hidden_case={
    "actual_issue_type": "login_access",
    "required_policy_keys": ["account_access", "password_reset"],
    "refund_eligible": None,
    "escalation_required": False,
    "customer_sentiment": "frustrated",
    "issue_resolved": False,
    "wrong_action_caused_damage": False,
    "correct_resolution_type": "reissue_reset_link",
    "correct_team": "account_access",
    "correct_priority": "medium",
    "correct_severity": None,
    "close_note": "Safe account-access guidance sent and reset link reissued.",
},

    @property
    def inbox_summary(self) -> str:
        return self.public_inbox_summary

    @property
    def open_questions(self) -> List[str]:
        return self.public_open_questions

TASKS: Dict[str, TaskSpec] = {
    "login_lockout": TaskSpec(
        task_id="login_lockout",
        title="Password-reset login lockout triage",
        difficulty="easy",
        public_inbox_summary=(
            "A customer says they still cannot log in after a password reset. "
            "They think the reset email may have gone to spam and they are unsure "
            "whether 2FA is also involved."
        ),
        public_open_questions=[
            "Who is the customer and what is the current account state?",
            "What guidance is safe and policy-compliant?",
            "Should any account-access action be taken before closing?",
        ],
        constraints=[
            "Do not ask for the customer's password.",
            "Do not suggest disabling 2FA globally.",
            "Use a calm, supportive tone.",
        ],
        customer_record={
            "customer_id": "cust_1001",
            "name": "Aarav Mehta",
            "email": "aarav.mehta@example.com",
            "region": "IN",
            "plan": "pro",
            "account_status": "active",
            "account_locked": False,
            "two_factor_enabled": True,
            "last_password_reset_at": "2026-04-06T08:10:00Z",
            "reset_email_delivery": "delivered",
        },
        order_record=None,
        payment_record=None,
        previous_tickets=[
            {
                "ticket_id": "t_2198",
                "created_at": "2026-03-18T10:15:00Z",
                "issue": "Password reset email not found",
                "resolution": "Customer found the reset mail in spam after resend.",
            }
        ],
        policy_docs={
            "account_access": (
                "For login or reset issues, agents must never ask for a password or OTP. "
                "Advise the customer to check spam/junk, resend the reset link if appropriate, "
                "and confirm access to a trusted 2FA device. Do not disable 2FA globally."
            ),
            "password_reset": (
                "If the account is active and not locked, a reset link may be reissued. "
                "The response should explain that email delivery delays or spam filtering can occur."
            ),
        },
        hidden_case={
            "actual_issue_type": "login_access",
            "required_policy_keys": ["account_access", "password_reset"],
            "refund_eligible": None,
            "escalation_required": False,
            "customer_sentiment": "frustrated",
            "issue_resolved": False,
            "wrong_action_caused_damage": False,
            "correct_resolution_type": "reissue_reset_link",
            "correct_team": "account_access",
            "correct_priority": "medium",
            "correct_severity": None,
            "close_note": "Safe account-access guidance sent and reset link reissued.",
        },
        expected_resolution={
            "team": "account_access",
            "priority": "medium",
            "status": "closed",
            "resolution_type": "reissue_reset_link",
            "required_tags": ["login", "password-reset"],
            "draft_must_include": [
                "reset link",
                "check spam",
                "trusted 2FA device",
            ],
            "close_note": "Safe account-access guidance sent and reset link reissued.",
        },
        required_actions=[
            "search_customer",
            "check_policy",
            "draft_response",
            "take_resolution_action",
            "close_ticket",
        ],
        required_revealed_sections=[
            "customer",
            "policy",
        ],
        min_meaningful_steps=4,
        forbidden_phrases=[
            "send me your password",
            "disable 2fa",
            "we cannot help",
        ],
        max_turns=6,
    ),
    "duplicate_charge_refund": TaskSpec(
        task_id="duplicate_charge_refund",
        title="Duplicate billing charge resolution",
        difficulty="medium",
        public_inbox_summary=(
            "A customer says they were charged twice for the same subscription renewal. "
            "They want the duplicate charge reversed."
        ),
        public_open_questions=[
            "Is there really a duplicate charge in the order/payment records?",
            "What refund policy applies here?",
            "What customer-facing timeline should be communicated?",
        ],
        constraints=[
            "Do not ask the customer to resend card details.",
            "Refunds must go to the original payment method.",
            "Set expectations that bank processing can take a few business days.",
        ],
        customer_record={
            "customer_id": "cust_2042",
            "name": "Elena Fischer",
            "email": "elena.fischer@example.eu",
            "region": "EU",
            "plan": "business",
            "account_status": "active",
        },
        order_record={
            "order_id": "ord_77881",
            "subscription_id": "sub_77881",
            "product": "Business Monthly",
            "renewal_date": "2026-04-04",
            "currency": "EUR",
            "amount": 49.00,
            "billing_status": "duplicate_charge_review_required",
        },
        payment_record={
            "payment_ids": ["pay_781", "pay_782"],
            "charge_count": 2,
            "duplicate_charge_detected": True,
            "charges": [
                {
                    "payment_id": "pay_781",
                    "amount": 49.00,
                    "currency": "EUR",
                    "status": "captured",
                    "captured_at": "2026-04-04T09:02:00Z",
                },
                {
                    "payment_id": "pay_782",
                    "amount": 49.00,
                    "currency": "EUR",
                    "status": "captured",
                    "captured_at": "2026-04-04T09:05:00Z",
                    "duplicate_of": "pay_781",
                },
            ],
        },
        previous_tickets=[
            {
                "ticket_id": "t_1987",
                "created_at": "2025-12-10T12:00:00Z",
                "issue": "Invoice copy request",
                "resolution": "Invoice shared.",
            }
        ],
        policy_docs={
            "billing_refunds": (
                "Confirmed duplicate subscription charges are refundable to the original payment method. "
                "Agents must not request full card numbers or alternate transfer details. "
                "Typical bank processing time is 3-5 business days after the refund is issued."
            ),
            "duplicate_charge": (
                "Before refunding, verify that two captured payments exist for the same renewal event. "
                "Mark the case for duplicate-charge review and note the source payment IDs."
            ),
        },
        hidden_case={
    "actual_issue_type": "duplicate_billing_charge",
    "required_policy_keys": ["billing_refunds", "duplicate_charge"],
    "refund_eligible": True,
    "escalation_required": False,
    "customer_sentiment": "frustrated",
    "issue_resolved": False,
    "wrong_action_caused_damage": False,
    "correct_resolution_type": "issue_refund",
    "correct_team": "billing",
    "correct_priority": "high",
    "correct_severity": None,
    "close_note": "Duplicate charge verified and refund issued to original payment method.",
},
        expected_resolution={
            "team": "billing",
            "priority": "high",
            "status": "closed",
            "resolution_type": "issue_refund",
            "required_tags": ["billing", "refund", "duplicate-charge"],
            "draft_must_include": [
                "refund",
                "original payment method",
                "3-5 business days",
            ],
            "close_note": "Duplicate charge verified and refund issued to original payment method.",
        },
        required_actions=[
            "search_customer",
            "view_order",
            "check_policy",
            "take_resolution_action",
            "draft_response",
            "close_ticket",
        ],
        required_revealed_sections=[
            "customer",
            "order",
            "payment",
            "policy",
        ],
        min_meaningful_steps=5,
        forbidden_phrases=[
            "send card number",
            "wire transfer",
            "immediate refund",
        ],
        max_turns=7,
    ),
    "eu_outage_incident": TaskSpec(
        task_id="eu_outage_incident",
        title="EU region outage incident coordination",
        difficulty="hard",
        public_inbox_summary=(
            "A service outage is affecting a portion of European users. "
            "Root cause is not yet confirmed, and a customer-facing update is needed."
        ),
        public_open_questions=[
            "Which team should own the active incident response?",
            "What severity should be assigned?",
            "What can be safely said publicly about ETA and workaround?",
        ],
        constraints=[
            "Do not claim the incident is fixed before verification.",
            "Do not promise an exact ETA.",
            "Mention a workaround only if one is actually available.",
            "Use incident-appropriate language for an ongoing outage.",
        ],
        customer_record={
            "customer_id": "tenant_eu_77",
            "tenant_name": "Northstar Analytics",
            "region": "EU",
            "affected_scope": "approximately 20% of EU users",
            "enterprise_tier": True,
        },
        order_record=None,
        payment_record=None,
        previous_tickets=[
            {
                "ticket_id": "inc_0441",
                "created_at": "2026-02-14T06:22:00Z",
                "issue": "Brief EU API latency spike",
                "resolution": "Recovered after failover. No persistent workaround was required.",
            }
        ],
        policy_docs={
            "incident_response": (
                "Active regional outages affecting a meaningful customer segment should be routed to incident_management. "
                "If customer impact is material but not total, sev2 is appropriate unless wider impact is confirmed."
            ),
            "public_comms": (
                "Public updates for active incidents must avoid exact ETAs unless confirmed by incident command. "
                "Use language such as investigating or monitoring. Mention the status page. "
                "Include a workaround only when one is known and verified."
            ),
        },
        hidden_case={
    "actual_issue_type": "regional_service_outage",
    "required_policy_keys": ["incident_response", "public_comms"],
    "refund_eligible": None,
    "escalation_required": True,
    "customer_sentiment": "urgent",
    "issue_resolved": False,
    "wrong_action_caused_damage": False,
    "correct_resolution_type": "publish_status_update",
    "correct_team": "incident_management",
    "correct_priority": "urgent",
    "correct_severity": "sev2",
    "close_note": "Incident escalated correctly and safe customer-facing update prepared.",
},
        expected_resolution={
            "team": "incident_management",
            "priority": "urgent",
            "severity": "sev2",
            "status": "handoff_complete",
            "resolution_type": "publish_status_update",
            "required_tags": ["incident", "outage", "eu"],
            "draft_must_include": [
                "status page",
                "no ETA",
                "workaround",
            ],
            "close_note": "Incident escalated correctly and safe customer-facing update prepared.",
            "workaround_available": True,
            "approved_workaround": "Customers may retry requests against the failover endpoint while mitigation is in progress.",
        },
        required_actions=[
            "search_customer",
            "check_policy",
            "escalate_case",
            "take_resolution_action",
            "draft_response",
            "close_ticket",
        ],
        required_revealed_sections=[
            "customer",
            "policy",
        ],
        min_meaningful_steps=5,
        forbidden_phrases=[
            "fixed",
            "resolved",
            "exact eta",
            "100% restored",
        ],
        max_turns=8,
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
        "inbox_summary": task.public_inbox_summary,
        "open_questions": task.public_open_questions,
        "constraints": task.constraints,
        "max_turns": task.max_turns,
        "allowed_actions": task.allowed_actions,
        "min_meaningful_steps": task.min_meaningful_steps,
    }


def public_tasks() -> dict:
    return {task_id: public_task_manifest(task) for task_id, task in TASKS.items()}