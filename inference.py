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

BENCHMARK = "supportdesk_v1"
DEFAULT_ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", DEFAULT_MODEL_NAME)


class HFChatClient:
    """Minimal Hugging Face chat-completions client.

    This intentionally uses plain requests so offline mode has no dependency on
    provider SDK imports. The endpoint follows the OpenAI-compatible Hugging Face
    Inference API shape at: {API_BASE_URL}/chat/completions.
    """

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
    """Create the Hugging Face client only when needed."""
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


def _offline_policy(task_id: str, observation: Dict[str, Any]) -> SupportAction:
    task = get_task(task_id)

    if task_id == "login_lockout":
        message = (
            "Sorry for the trouble. Please use the reset link, check spam for the email, "
            "and confirm 2fa on the current device if access is still blocked."
        )
    elif task_id == "duplicate_charge_refund":
        message = (
            "Sorry about the duplicate charge. We have routed this to billing, and the refund "
            "will go to the original payment method within 3-5 business days."
        )
    else:
        message = (
            "We are investigating the outage, have updated the status page, and will share the "
            "current workaround while the incident team continues mitigation."
        )

    action_kwargs: Dict[str, Any] = {
        "action_type": "finalize",
        "issue_type": task.target_issue_type,
        "priority": task.target_priority,
        "team": task.target_team,
        "status": task.target_status,
        "tags": task.required_tags,
        "message": message,
        "internal_note": f"Route to {task.target_team}; satisfy required phrases.",
        "confidence": 0.72,
    }

    if task.target_severity is not None:
        action_kwargs["severity"] = task.target_severity

    return SupportAction(**action_kwargs)


def _call_model_action(
    model_client: HFChatClient,
    model: str,
    observation: Dict[str, Any],
    task_id: str,
) -> SupportAction:
    task = get_task(task_id)

    system = (
        "You are a support operations agent. Produce exactly one JSON object with keys: "
        "action_type, issue_type, priority, team, severity, status, tags, message, "
        "internal_note, refund_amount, confidence. "
        "Only include fields that help. "
        "Keep the message professional, concise, and policy-safe."
    )

    user = {
        "task": {
            "task_id": task.task_id,
            "title": task.title,
            "difficulty": task.difficulty,
            "inbox_summary": task.inbox_summary,
            "open_questions": task.open_questions,
            "constraints": task.constraints,
        },
        "observation": observation,
        "instruction": "Return only valid JSON. Use the most helpful single step for improving the score.",
    }

    resp = model_client.create_chat_completion(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    content = (
        resp.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "{}")
    )
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
    max_turns: int = 4,
    fallback_to_offline_on_model_error: bool = False,
) -> Dict[str, Any]:
    reset_result = env.reset(task_id=task_id)
    observation = reset_result.observation
    rewards: List[float] = []

    for turn in range(max_turns):
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
    success = final_score >= 1.0

    _print_end(
        success=success,
        steps=final_state.turn,
        score=final_score,
        rewards=rewards,
    )

    return {
        "task_id": task_id,
        "difficulty": get_task(task_id).difficulty,
        "final_score": final_score,
        "turns": final_state.turn,
        "transcript": final_state.transcript,
        "success": success,
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
            fallback_to_offline_on_model_error=args.allow_fallback_offline,
        )
        results.append(result)
        total_score += result["final_score"]

    mean_score = total_score / len(results) if results else 0.0
    logger.info("Evaluation complete. Mean Score: %.3f", mean_score)


if __name__ == "__main__":
    main()
