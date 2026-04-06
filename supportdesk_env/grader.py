from __future__ import annotations

from typing import Any, Dict, List

from .models import SupportAction
from .task_bank import TaskSpec


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _safe_tags(tags: List[str] | None) -> List[str]:
    if not tags:
        return []
    return [t.strip().lower() for t in tags if t and t.strip()]


def draft_keyword_hits(task: TaskSpec, message: str | None) -> Dict[str, Any]:
    """
    Lightweight helper for checking whether a customer-facing draft
    contains the expected phrases for the task.
    """
    text = _norm(message)
    required = [p.lower() for p in task.expected_resolution.get("draft_must_include", [])]

    hits: List[str] = []
    missing: List[str] = []

    for phrase in required:
        if phrase == "no eta":
            if (
                "no eta" in text
                or "do not have an eta" in text
                or "don't have an eta" in text
                or "dont have an eta" in text
            ):
                hits.append(phrase)
            else:
                missing.append(phrase)
        elif phrase == "workaround":
            if "workaround" in text or "failover endpoint" in text:
                hits.append(phrase)
            else:
                missing.append(phrase)
        elif phrase in text:
            hits.append(phrase)
        else:
            missing.append(phrase)

    ratio = 1.0 if not required else len(hits) / len(required)
    return {
        "ratio": ratio,
        "hits": hits,
        "missing": missing,
    }


def forbidden_phrase_hits(task: TaskSpec, message: str | None) -> List[str]:
    """
    Return forbidden phrases found in a draft or note.
    """
    text = _norm(message)
    return [phrase for phrase in task.forbidden_phrases if phrase.lower() in text]


def resolution_alignment(task: TaskSpec, action: SupportAction) -> Dict[str, Any]:
    """
    Check whether a take_resolution_action payload aligns with the task expectation.
    """
    expected = task.expected_resolution
    expected_type = expected.get("resolution_type")
    expected_tags = set(t.lower() for t in expected.get("required_tags", []))
    actual_tags = set(_safe_tags(action.tags))

    resolution_type_match = action.resolution_type == expected_type
    tag_match_ratio = 1.0 if not expected_tags else len(expected_tags & actual_tags) / len(expected_tags)

    return {
        "expected_resolution_type": expected_type,
        "actual_resolution_type": action.resolution_type,
        "resolution_type_match": resolution_type_match,
        "tag_match_ratio": tag_match_ratio,
        "matched_tags": sorted(expected_tags & actual_tags),
        "missing_tags": sorted(expected_tags - actual_tags),
    }


def escalation_alignment(task: TaskSpec, action: SupportAction) -> Dict[str, Any]:
    """
    Check whether an escalation action matches the expected team / priority / severity.
    """
    expected = task.expected_resolution

    return {
        "team_match": _norm(action.team) == _norm(expected.get("team")),
        "priority_match": _norm(action.priority) == _norm(expected.get("priority")),
        "severity_match": _norm(action.severity) == _norm(expected.get("severity"))
        if expected.get("severity") is not None
        else action.severity is None,
        "expected_team": expected.get("team"),
        "expected_priority": expected.get("priority"),
        "expected_severity": expected.get("severity"),
    }


def required_actions_status(task: TaskSpec, action_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize which required actions have already been taken.
    """
    taken = {
        entry.get("action", {}).get("action_type")
        for entry in action_history
        if isinstance(entry, dict)
    }

    required = [a for a in task.required_actions if a != "close_ticket"]
    missing = [a for a in required if a not in taken]

    return {
        "required": required,
        "taken": sorted(a for a in taken if a),
        "missing": missing,
        "all_required_done": len(missing) == 0,
    }


def build_close_readiness_report(
    task: TaskSpec,
    *,
    revealed_sections: List[str],
    meaningful_steps: List[str],
    draft_message: str | None,
    violations: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Helper report for whether a case is structurally ready to close.
    This does not replace the environment's final close check, but keeps
    the logic reusable and testable.
    """
    violations = violations or []
    revealed = set(revealed_sections)
    required_sections = set(task.required_revealed_sections)

    draft_report = draft_keyword_hits(task, draft_message)
    forbidden_hits = forbidden_phrase_hits(task, draft_message)

    missing: List[str] = []

    if len(set(meaningful_steps)) < task.min_meaningful_steps:
        missing.append(
            f"need at least {task.min_meaningful_steps} meaningful steps"
        )

    for section in sorted(required_sections):
        if section not in revealed:
            missing.append(f"required data section '{section}' not revealed")

    if not draft_message:
        missing.append("draft response missing")
    elif draft_report["missing"]:
        missing.append("draft response is missing required customer-facing details")

    if forbidden_hits or violations:
        missing.append("draft or case history contains policy-risky language")

    return {
        "can_close_structurally": len(missing) == 0,
        "missing": missing,
        "draft_hits": draft_report["hits"],
        "draft_missing": draft_report["missing"],
        "forbidden_hits": forbidden_hits,
    }


def summarize_action(action: SupportAction) -> Dict[str, Any]:
    return action.model_dump(exclude_none=True)