from __future__ import annotations

from supportdesk_env.models import SupportAction
from supportdesk_env.server.environment import SupportDeskEnvironment
from fastapi.testclient import TestClient
from supportdesk_env.server.app import APP_STATE, app


def test_reset_starts_with_partial_information() -> None:
    env = SupportDeskEnvironment()
    result = env.reset(task_id="duplicate_charge_refund")

    obs = result.observation
    assert result.done is False
    assert obs.task_id == "duplicate_charge_refund"
    assert obs.turn == 0
    assert obs.can_close is False
    assert "customer" not in obs.revealed_data
    assert "order" not in obs.revealed_data
    assert "payment" not in obs.revealed_data
    assert "policy" not in obs.revealed_data
    assert len(obs.available_actions) >= 5


def test_cannot_close_immediately_after_reset() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="login_lockout")

    result = env.step(SupportAction(action_type="close_ticket", confidence=0.2))

    assert result.done is False
    assert result.reward.total < 0
    assert "Cannot close yet" in result.observation.last_feedback
    assert result.observation.can_close is False
    missing = result.info.get("missing_requirements", [])
    assert any("meaningful steps" in item for item in missing)
    assert any("search_customer" in item or "customer" in item for item in missing)


def test_search_customer_reveals_customer_section() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="login_lockout")

    result = env.step(
        SupportAction(
            action_type="search_customer",
            query="login reset issue customer lookup",
            confidence=0.8,
        )
    )

    obs = result.observation
    assert result.done is False
    assert result.reward.total > 0
    assert "customer" in obs.revealed_data
    assert obs.revealed_data["customer"]["customer_id"] == "cust_1001"
    assert obs.turn == 1


def test_view_order_reveals_order_and_payment_sections() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="duplicate_charge_refund")

    env.step(
        SupportAction(
            action_type="search_customer",
            query="duplicate subscription charge customer lookup",
            confidence=0.7,
        )
    )
    result = env.step(
        SupportAction(
            action_type="view_order",
            customer_id="cust_2042",
            confidence=0.85,
        )
    )

    obs = result.observation
    assert "order" in obs.revealed_data
    assert "payment" in obs.revealed_data
    assert obs.revealed_data["order"]["order_id"] == "ord_77881"
    assert obs.revealed_data["payment"]["duplicate_charge_detected"] is True


def test_repeated_info_reveal_action_gets_penalized() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="login_lockout")

    first = env.step(
        SupportAction(
            action_type="search_customer",
            query="login reset issue customer lookup",
            confidence=0.8,
        )
    )
    second = env.step(
        SupportAction(
            action_type="search_customer",
            query="repeat lookup",
            confidence=0.5,
        )
    )

    assert first.reward.total > 0
    assert second.reward.total < 0
    assert "already inspected" in second.observation.last_feedback.lower()


def test_draft_without_message_is_invalid() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="login_lockout")

    result = env.step(
        SupportAction(
            action_type="draft_response",
            confidence=0.4,
        )
    )

    assert result.done is False
    assert result.reward.total < 0
    assert "requires a non-empty customer-facing message" in result.observation.last_feedback


def test_duplicate_charge_wrong_short_path_does_not_pass() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="duplicate_charge_refund")

    env.step(
        SupportAction(
            action_type="draft_response",
            message="We will check this.",
            confidence=0.4,
        )
    )
    result = env.step(SupportAction(action_type="close_ticket", confidence=0.2))

    assert result.done is False
    assert result.reward.total < 0
    assert result.observation.can_close is False
    assert "Cannot close yet" in result.observation.last_feedback


def test_duplicate_charge_happy_path_closes_successfully() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="duplicate_charge_refund")

    r1 = env.step(
        SupportAction(
            action_type="search_customer",
            query="duplicate subscription charge customer lookup",
            confidence=0.8,
        )
    )
    assert r1.done is False

    r2 = env.step(
        SupportAction(
            action_type="view_order",
            customer_id="cust_2042",
            confidence=0.85,
        )
    )
    assert "order" in r2.observation.revealed_data
    assert "payment" in r2.observation.revealed_data

    r3 = env.step(
        SupportAction(
            action_type="check_policy",
            policy_key="billing_refunds",
            confidence=0.9,
        )
    )
    assert "policy" in r3.observation.revealed_data

    r4 = env.step(
        SupportAction(
            action_type="take_resolution_action",
            resolution_type="issue_refund",
            status="completed",
            tags=["billing", "refund", "duplicate-charge"],
            resolution_payload={
                "order_id": "ord_77881",
                "duplicate_payment_id": "pay_782",
                "refund_destination": "original_payment_method",
            },
            internal_note="Verified duplicate payment and issued refund.",
            confidence=0.95,
        )
    )
    assert r4.done is False

    r5 = env.step(
        SupportAction(
            action_type="draft_response",
            message=(
                "Sorry about the duplicate charge. I verified the duplicate payment and have issued a refund "
                "to your original payment method. Most banks reflect the refund within 3-5 business days."
            ),
            confidence=0.95,
        )
    )
    assert r5.observation.can_close is True

    r6 = env.step(
        SupportAction(
            action_type="close_ticket",
            confidence=0.98,
        )
    )

    assert r6.done is True
    assert r6.reward.total > 0
    assert "Duplicate charge verified" in r6.observation.last_feedback or "workflow completion" in r6.observation.last_feedback


def test_outage_requires_escalation_before_close() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="eu_outage_incident")

    env.step(
        SupportAction(
            action_type="search_customer",
            query="eu outage affected tenant lookup",
            confidence=0.7,
        )
    )
    env.step(
        SupportAction(
            action_type="check_policy",
            policy_key="incident_response",
            confidence=0.8,
        )
    )
    env.step(
        SupportAction(
            action_type="take_resolution_action",
            resolution_type="publish_status_update",
            status="handoff_complete",
            tags=["incident", "outage", "eu"],
            resolution_payload={
                "status_page": True,
                "workaround": "Customers may retry requests against the failover endpoint while mitigation is in progress.",
            },
            confidence=0.8,
        )
    )
    env.step(
        SupportAction(
            action_type="draft_response",
            message=(
                "We are actively investigating the EU service disruption and have posted an update on the status page. "
                "We do not have an ETA right now. As a temporary workaround, customers may retry requests against the "
                "failover endpoint while mitigation is in progress."
            ),
            confidence=0.9,
        )
    )
    result = env.step(SupportAction(action_type="close_ticket", confidence=0.5))

    assert result.done is False
    assert result.reward.total < 0
    missing = result.info.get("missing_requirements", [])
    assert any("escalate_case" in item or "escalation" in item for item in missing)


def test_outage_happy_path_closes_successfully() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="eu_outage_incident")

    env.step(
        SupportAction(
            action_type="search_customer",
            query="eu outage affected tenant lookup",
            confidence=0.8,
        )
    )
    env.step(
        SupportAction(
            action_type="check_policy",
            policy_key="incident_response",
            confidence=0.85,
        )
    )
    env.step(
        SupportAction(
            action_type="escalate_case",
            team="incident_management",
            priority="urgent",
            severity="sev2",
            status="handoff_in_progress",
            tags=["incident", "outage", "eu"],
            internal_note="Material EU impact observed; routing to incident management.",
            confidence=0.95,
        )
    )
    env.step(
        SupportAction(
            action_type="take_resolution_action",
            resolution_type="publish_status_update",
            status="handoff_complete",
            tags=["incident", "outage", "eu"],
            resolution_payload={
                "status_page": True,
                "workaround": "Customers may retry requests against the failover endpoint while mitigation is in progress.",
            },
            internal_note="Published safe incident update with workaround and no exact ETA.",
            confidence=0.95,
        )
    )
    r5 = env.step(
        SupportAction(
            action_type="draft_response",
            message=(
                "We are actively investigating the EU service disruption and have posted an update on the status page. "
                "We do not have an ETA right now. As a temporary workaround, customers may retry requests against the "
                "failover endpoint while mitigation is in progress."
            ),
            confidence=0.95,
        )
    )

    assert r5.observation.can_close is True

    r6 = env.step(
        SupportAction(
            action_type="close_ticket",
            confidence=0.99,
        )
    )

    assert r6.done is True
    assert r6.reward.total > 0


def test_turn_limit_ends_incomplete_episode() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="login_lockout")

    result = None
    for _ in range(6):
        result = env.step(
            SupportAction(
                action_type="draft_response",
                message="Checking this for you.",
                confidence=0.3,
            )
        )

    assert result is not None
    assert result.done is True
    assert result.info["done_reason"] == "max_turns"

def test_public_state_redacts_hidden_ticket_truth() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="duplicate_charge_refund")

    public_state = env.public_state
    dumped = public_state.model_dump()

    forbidden_keys = {
        "hidden_state",
        "transcript",
        "violations",
        "verification_state",
        "resolution_state",
        "escalation_state",
        "draft_message",
        "last_action",
        "last_observation",
        "meaningful_steps",
    }

    for key in forbidden_keys:
        assert key not in dumped


def test_public_state_does_not_expose_private_ticket_answers_after_progress() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="eu_outage_incident")

    env.step(
        SupportAction(
            action_type="search_customer",
            query="eu outage affected tenant lookup",
            confidence=0.8,
        )
    )
    env.step(
        SupportAction(
            action_type="check_policy",
            policy_key="incident_response",
            confidence=0.85,
        )
    )

    public_state = env.public_state
    dumped = public_state.model_dump()
    text = str(dumped)

    forbidden_strings = [
        "actual_issue_type",
        "required_policy_keys",
        "refund_eligible",
        "escalation_required",
        "customer_sentiment",
        "issue_resolved",
        "wrong_action_caused_damage",
        "correct_resolution_type",
        "correct_team",
        "correct_priority",
        "correct_severity",
    ]

    for item in forbidden_strings:
        assert item not in text


def test_close_feedback_does_not_leak_exact_target_answer() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="duplicate_charge_refund")

    result = env.step(SupportAction(action_type="close_ticket", confidence=0.2))

    assert result.done is False
    assert result.reward.total < 0

    feedback = result.observation.last_feedback
    missing = result.info.get("missing_requirements", [])
    combined = feedback + " || " + " || ".join(missing)

    forbidden_strings = [
        "issue_refund",
        "incident_management",
        "sev2",
        "urgent",
        "reissue_reset_link",
        "publish_status_update",
    ]

    for item in forbidden_strings:
        assert item not in combined


def test_wrong_action_damage_is_tracked_privately_not_publicly() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="login_lockout")

    result = env.step(
        SupportAction(
            action_type="take_resolution_action",
            resolution_type="issue_refund",
            status="completed",
            tags=["billing", "refund"],
            resolution_payload={"reason": "goodwill"},
            confidence=0.8,
        )
    )

    assert result.done is False
    assert env.state.hidden_state.wrong_action_caused_damage is True

    public_state = env.public_state.model_dump()
    public_text = str(public_state)

    assert "wrong_action_caused_damage" not in public_text
    assert "hidden_state" not in public_text


def test_public_state_shows_progress_without_exposing_solution() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id="duplicate_charge_refund")

    env.step(
        SupportAction(
            action_type="search_customer",
            query="duplicate subscription charge customer lookup",
            confidence=0.8,
        )
    )
    env.step(
        SupportAction(
            action_type="view_order",
            customer_id="cust_2042",
            confidence=0.85,
        )
    )

    public_state = env.public_state

    assert public_state.task_id == "duplicate_charge_refund"
    assert public_state.turn == 2
    assert public_state.draft_exists is False
    assert public_state.escalation_recorded is False
    assert public_state.resolution_recorded is False
    assert "customer" in public_state.revealed_data
    assert "order" in public_state.revealed_data
    assert "payment" in public_state.revealed_data