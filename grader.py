from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, Tuple

from .models import SupportAction
from .task_bank import TaskSpec


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _keyword_fraction(text: str | None, keywords: list[str]) -> Tuple[float, list[str]]:
    haystack = _norm(text)
    if not keywords:
        return 1.0, []
    matches = [kw for kw in keywords if kw.lower() in haystack]
    return len(matches) / len(keywords), matches


def _safe_tags(tags: list[str]) -> list[str]:
    return [t.strip().lower() for t in tags if t and t.strip()]


def score_action(task: TaskSpec, action: SupportAction) -> Dict[str, Any]:
    """Return a deterministic rubric score and per-criterion breakdown."""

    criteria: Dict[str, float] = {}
    violations: list[str] = []

    # Core routing fields
    criteria["team"] = 1.0 if _norm(action.team) == task.target_team else 0.0
    criteria["priority"] = 1.0 if _norm(action.priority) == task.target_priority else 0.0
    criteria["issue_type"] = 1.0 if _norm(action.issue_type) == task.target_issue_type else 0.0

    if task.target_severity is None:
        criteria["severity"] = 1.0 if action.severity is None else 0.5
    else:
        criteria["severity"] = 1.0 if _norm(action.severity) == task.target_severity else 0.0

    # Operational status and tags
    criteria["status"] = 1.0 if _norm(action.status) == task.target_status else 0.0

    tags = set(_safe_tags(action.tags))
    required_tags = set(task.required_tags)
    if required_tags:
        criteria["tags"] = len(tags & required_tags) / len(required_tags)
    else:
        criteria["tags"] = 1.0

    # Message quality / policy compliance
    message_fraction, matched_phrases = _keyword_fraction(action.message, task.required_phrases)
    criteria["message_phrases"] = message_fraction

    message_text = _norm(action.message)
    forbidden_hits = [phrase for phrase in task.forbidden_phrases if phrase.lower() in message_text]
    if forbidden_hits:
        violations.extend(f"forbidden_phrase:{phrase}" for phrase in forbidden_hits)

    internal_note_fraction, _ = _keyword_fraction(action.internal_note, task.required_phrases[:2])
    criteria["internal_note"] = internal_note_fraction

    # A gentle incentive to keep the action type aligned with the phase of work.
    if task.difficulty == "easy":
        action_type_bonus = 1.0 if action.action_type in {"classify", "draft_reply", "finalize"} else 0.4
    elif task.difficulty == "medium":
        action_type_bonus = 1.0 if action.action_type in {"draft_reply", "resolve", "finalize"} else 0.5
    else:
        action_type_bonus = 1.0 if action.action_type in {"escalate", "resolve", "finalize"} else 0.5
    criteria["action_type"] = action_type_bonus

    if action.confidence < 0.2:
        violations.append("low_confidence")

    weights: Dict[str, float] = {
        "team": 0.18,
        "priority": 0.14,
        "issue_type": 0.12,
        "severity": 0.08,
        "status": 0.08,
        "tags": 0.10,
        "message_phrases": 0.22,
        "internal_note": 0.03,
        "action_type": 0.05,
    }

    raw = sum(criteria[name] * weight for name, weight in weights.items())

    # Penalties for clearly bad behavior.
    penalty = 0.0
    if forbidden_hits:
        penalty += min(0.25, 0.08 * len(forbidden_hits))
    if action.refund_amount is not None and action.refund_amount > 0 and task.task_id != "duplicate_charge_refund":
        penalty += 0.08
    if action.severity and task.target_severity is None:
        penalty += 0.03
    if action.action_type == "finalize" and raw < 0.75:
        penalty += 0.05

    score = max(0.0, min(1.0, raw - penalty))
    explanation = _build_explanation(task, criteria, violations, matched_phrases)
    return {
        "score": score,
        "criteria": criteria,
        "penalty": penalty,
        "violations": violations,
        "explanation": explanation,
    }


def _build_explanation(
    task: TaskSpec,
    criteria: Dict[str, float],
    violations: list[str],
    matched_phrases: list[str],
) -> str:
    top_hits = []
    if criteria.get("team", 0) >= 1.0:
        top_hits.append("team")
    if criteria.get("priority", 0) >= 1.0:
        top_hits.append("priority")
    if criteria.get("issue_type", 0) >= 1.0:
        top_hits.append("issue type")
    if criteria.get("message_phrases", 0) > 0:
        top_hits.append(f"phrases:{', '.join(matched_phrases[:3])}")

    bits = []
    if top_hits:
        bits.append("matched " + "; ".join(top_hits))
    if violations:
        bits.append("violations: " + ", ".join(violations[:3]))
    if not bits:
        bits.append("partial progress only")
    return f"{task.title}: " + " | ".join(bits)


def build_feedback(task: TaskSpec, rubric: Dict[str, Any]) -> str:
    criteria = rubric["criteria"]
    lines = []
    if criteria.get("team", 0) < 1.0:
        lines.append(f"route to {task.target_team}")
    if criteria.get("priority", 0) < 1.0:
        lines.append(f"set priority {task.target_priority}")
    if criteria.get("issue_type", 0) < 1.0:
        lines.append(f"label the issue as {task.target_issue_type}")
    if task.target_severity and criteria.get("severity", 0) < 1.0:
        lines.append(f"set severity {task.target_severity}")
    if criteria.get("tags", 0) < 1.0:
        lines.append("apply the required tags")
    if criteria.get("message_phrases", 0) < 1.0:
        lines.append("include the required customer-facing phrases")
    if rubric["violations"]:
        lines.append("remove forbidden language")
    if not lines:
        lines.append("good progress; finalize the case cleanly")
    return "; ".join(lines)


def summarize_action(action: SupportAction) -> Dict[str, Any]:
    return action.model_dump()
