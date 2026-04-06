from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from supportdesk_env.client import SupportDeskClient
from supportdesk_env.logging_config import get_logger
from supportdesk_env.models import SupportAction
from supportdesk_env.task_bank import TASK_ORDER, get_task

load_dotenv()

logger = get_logger("inference")

BENCHMARK = "supportdesk_v2"
DEFAULT_ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", DEFAULT_MODEL_NAME)


class HFChatClient:
    """Minimal Hugging Face chat-completions client."""

    def __init__(self, api_base_url: str, token: str, timeout: float = 60.0) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self.session.post(
            f"{self.api_base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def build_model_client(api_base_url: str | None = None) -> HFChatClient:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required unless --offline is set.")
    return HFChatClient(api_base_url=api_base_url or DEFAULT_API_BASE_URL, token=token)


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_compact(v) for v in value if v is not None]
    return value


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for candidate in reversed(fenced):
        try:
            return json.loads(candidate)
        except Exception:
            continue

    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError(f"Could not parse JSON from model output: {text[:400]}")


def _history_action_types(observation: Dict[str, Any]) -> List[str]:
    return [
        str(entry.get("action_type"))
        for entry in observation.get("action_history", [])
        if entry.get("action_type")
    ]


def _revealed(observation: Dict[str, Any]) -> Dict[str, Any]:
    return observation.get("revealed_data", {}) or {}


def _login_lockout_action(observation: Dict[str, Any]) -> SupportAction:
    revealed = _revealed(observation)

    if observation.get("can_close"):
        return SupportAction(action_type="close_ticket", confidence=0.95)

    if "customer" not in revealed:
        return SupportAction(
            action_type="search_customer",
            query="login reset issue customer lookup",
            confidence=0.75,
        )

    if "policy" not in revealed:
        return SupportAction(
            action_type="check_policy",
            policy_key="account_access",
            confidence=0.80,
        )

    if not revealed.get("previous_tickets"):
        return SupportAction(
            action_type="inspect_previous_tickets",
            confidence=0.62,
        )

    history = _history_action_types(observation)
    if "take_resolution_action" not in history:
        return SupportAction(
            action_type="take_resolution_action",
            resolution_type="reissue_reset_link",
            status="completed",
            tags=["login", "password-reset"],
            resolution_payload={
                "channel": "email",
                "reason": "customer still cannot log in after reset attempt",
            },
            internal_note="Reissued reset link after checking account-access guidance.",
            confidence=0.86,
        )

    if "draft_response" not in history:
        return SupportAction(
            action_type="draft_response",
            message=(
                "Sorry for the trouble. I have reissued your reset link. "
                "Please check your spam or junk folder if you do not see it right away, "
                "and make sure you have access to your trusted 2FA device when signing in. "
                "For security, we do not need your password or OTP."
            ),
            confidence=0.88,
        )

    return SupportAction(action_type="close_ticket", confidence=0.93)


def _duplicate_charge_action(observation: Dict[str, Any]) -> SupportAction:
    revealed = _revealed(observation)

    if observation.get("can_close"):
        return SupportAction(action_type="close_ticket", confidence=0.96)

    if "customer" not in revealed:
        return SupportAction(
            action_type="search_customer",
            query="duplicate subscription charge customer lookup",
            confidence=0.78,
        )

    if "order" not in revealed or "payment" not in revealed:
        customer = revealed.get("customer", {})
        return SupportAction(
            action_type="view_order",
            customer_id=customer.get("customer_id"),
            confidence=0.84,
        )

    if "policy" not in revealed:
        return SupportAction(
            action_type="check_policy",
            policy_key="billing_refunds",
            confidence=0.84,
        )

    history = _history_action_types(observation)
    if "take_resolution_action" not in history:
        payment = revealed.get("payment", {})
        order = revealed.get("order", {})
        duplicate_payment_id = None
        for charge in payment.get("charges", []):
            if charge.get("duplicate_of"):
                duplicate_payment_id = charge.get("payment_id")
                break

        return SupportAction(
            action_type="take_resolution_action",
            resolution_type="issue_refund",
            status="completed",
            tags=["billing", "refund", "duplicate-charge"],
            resolution_payload={
                "order_id": order.get("order_id"),
                "duplicate_payment_id": duplicate_payment_id,
                "refund_destination": "original_payment_method",
            },
            internal_note="Verified duplicate captured payment and issued refund to original payment method.",
            confidence=0.90,
        )

    if "draft_response" not in history:
        return SupportAction(
            action_type="draft_response",
            message=(
                "Sorry about the duplicate charge. I verified the duplicate payment and have issued a refund "
                "to your original payment method. Most banks reflect the refund within 3-5 business days."
            ),
            confidence=0.90,
        )

    return SupportAction(action_type="close_ticket", confidence=0.94)


def _eu_outage_action(observation: Dict[str, Any]) -> SupportAction:
    revealed = _revealed(observation)

    if observation.get("can_close"):
        return SupportAction(action_type="close_ticket", confidence=0.96)

    if "customer" not in revealed:
        return SupportAction(
            action_type="search_customer",
            query="eu outage affected tenant lookup",
            confidence=0.78,
        )

    if "policy" not in revealed:
        return SupportAction(
            action_type="check_policy",
            policy_key="incident_response",
            confidence=0.83,
        )

    history = _history_action_types(observation)
    if "escalate_case" not in history:
        return SupportAction(
            action_type="escalate_case",
            team="incident_management",
            priority="urgent",
            severity="sev2",
            status="handoff_in_progress",
            tags=["incident", "outage", "eu"],
            internal_note="Material EU impact observed; routing to incident management as sev2.",
            confidence=0.91,
        )

    if "take_resolution_action" not in history:
        return SupportAction(
            action_type="take_resolution_action",
            resolution_type="publish_status_update",
            status="handoff_complete",
            tags=["incident", "outage", "eu"],
            resolution_payload={
                "status_page": True,
                "workaround": "Customers may retry requests against the failover endpoint while mitigation is in progress.",
            },
            internal_note="Published safe incident update with approved workaround and no exact ETA.",
            confidence=0.90,
        )

    if "draft_response" not in history:
        return SupportAction(
            action_type="draft_response",
            message=(
                "We are actively investigating the EU service disruption and have posted an update on the status page. "
                "We do not have an ETA right now. As a temporary workaround, customers may retry requests against the "
                "failover endpoint while mitigation is in progress."
            ),
            confidence=0.91,
        )

    return SupportAction(action_type="close_ticket", confidence=0.94)


def _offline_policy(task_id: str, observation: Dict[str, Any]) -> SupportAction:
    if task_id == "login_lockout":
        return _login_lockout_action(observation)
    if task_id == "duplicate_charge_refund":
        return _duplicate_charge_action(observation)
    return _eu_outage_action(observation)


def _call_model_action(
    model_client: HFChatClient,
    model: str,
    observation: Dict[str, Any],
    task_id: str,
) -> SupportAction:
    task = get_task(task_id)

    system = (
        "You are a support operations agent acting inside a multi-step environment. "
        "You do NOT know all facts at reset. "
        "Choose exactly one next operational action that best improves progress based only on the current observation. "
        "Valid action_type values are: search_customer, view_order, check_policy, "
        "inspect_previous_tickets, draft_response, escalate_case, take_resolution_action, close_ticket. "
        "Do not jump to close_ticket unless the observation says can_close=true. "
        "Return exactly one valid JSON object matching the action schema. No markdown. No explanation."
    )

    user = {
        "task": {
            "task_id": task.task_id,
            "title": task.title,
            "difficulty": task.difficulty,
            "inbox_summary": task.public_inbox_summary,
            "open_questions": task.public_open_questions,
            "constraints": task.constraints,
            "allowed_actions": task.allowed_actions,
        },
        "observation": observation,
        "instruction": (
            "Pick the single best next action. "
            "Use currently revealed_data and action_history. "
            "If key information is still missing, inspect before acting. "
            "If a response is drafted, keep it concise and policy-safe."
        ),
    }

    resp = model_client.create_chat_completion(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        max_tokens=350,
    )

    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    data = _extract_json(content)
    return SupportAction.model_validate(data)


def _print_start(task: str, model_name: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model_name}", flush=True)


def _print_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def _print_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _summarize_action(action: SupportAction) -> str:
    data = action.model_dump(exclude_none=True)
    return json.dumps(_compact(data), ensure_ascii=False, separators=(",", ":"))


def run_episode(
    env: SupportDeskClient,
    task_id: str,
    use_offline: bool,
    model: str,
    model_client: Optional[HFChatClient] = None,
    max_turns: Optional[int] = None,
    fallback_to_offline_on_model_error: bool = False,
) -> Dict[str, Any]:
    task = get_task(task_id)
    turn_budget = max_turns or task.max_turns

    reset_result = env.reset(task_id=task_id)
    observation = reset_result.observation
    rewards: List[float] = []
    last_done_reason = "continue"

    for turn in range(turn_budget):
        obs_dict = observation.model_dump()

        if use_offline:
            action = _offline_policy(task_id, obs_dict)
        else:
            if model_client is None:
                raise RuntimeError("model_client is required when use_offline=False")

            try:
                action = _call_model_action(model_client, model, obs_dict, task_id)
            except Exception as exc:
                if fallback_to_offline_on_model_error:
                    logger.warning("Model call failed; falling back to offline policy: %s", exc)
                    action = _offline_policy(task_id, obs_dict)
                else:
                    raise RuntimeError(f"Model call failed for task {task_id}: {exc}") from exc

        step_result = env.step(action)
        observation = step_result.observation
        reward = step_result.reward.total
        done = step_result.done
        error = step_result.info.get("error")
        last_done_reason = str(step_result.info.get("done_reason", "continue"))

        rewards.append(reward)
        _print_step(
            step=turn + 1,
            action=_summarize_action(action),
            reward=reward,
            done=done,
            error=error,
        )

        if done:
            break

    final_state = env.state()
    final_score = final_state.best_score
    success = last_done_reason == "closed"

    _print_end(
        success=success,
        steps=final_state.turn,
        score=final_score,
        rewards=rewards,
    )

    return {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "final_score": final_score,
        "turns": final_state.turn,
        "transcript": final_state.transcript,
        "success": success,
        "done_reason": last_done_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for SupportDesk OpenEnv")
    parser.add_argument("--env-url", default=DEFAULT_ENV_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--offline", action="store_true", help="Use a deterministic heuristic instead of the model.")
    parser.add_argument(
        "--allow-fallback-offline",
        action="store_true",
        help="If model inference fails, fall back to offline heuristic.",
    )
    parser.add_argument("--tasks", nargs="*", default=TASK_ORDER, help="Task ids to run.")
    parser.add_argument("--max-turns", type=int, default=None, help="Optional override for turn budget per task.")
    args = parser.parse_args()

    env = SupportDeskClient(base_url=args.env_url)

    model_client: Optional[HFChatClient] = None
    model_name_for_logs = args.model if not args.offline else "offline-heuristic"

    if not args.offline:
        try:
            model_client = build_model_client(DEFAULT_API_BASE_URL)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc

    results: List[Dict[str, Any]] = []
    total_score = 0.0

    for task_id in args.tasks:
        _print_start(task_id, model_name_for_logs)
        result = run_episode(
            env=env,
            task_id=task_id,
            use_offline=args.offline,
            model=args.model,
            model_client=model_client,
            max_turns=args.max_turns,
            fallback_to_offline_on_model_error=args.allow_fallback_offline,
        )
        results.append(result)
        total_score += result["final_score"]

    mean_score = total_score / len(results) if results else 0.0
    logger.info("Evaluation complete. Mean Score: %.3f", mean_score)


if __name__ == "__main__":
    main()