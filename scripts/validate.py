from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supportdesk_env.logging_config import get_logger
from supportdesk_env.models import SupportAction
from supportdesk_env.server.environment import SupportDeskEnvironment
from supportdesk_env.task_bank import TASK_ORDER, TASKS

logger = get_logger("validation")


def main() -> None:
    """Automated validation of all tasks in the bank."""
    env = SupportDeskEnvironment()
    logger.info("Validating %s tasks...", len(TASK_ORDER))

    for task_id in TASK_ORDER:
        reset_result = env.reset(task_id=task_id)
        if reset_result.observation.task_id != task_id:
            logger.error("Failed to reset task: %s", task_id)
            raise SystemExit(1)

        action = SupportAction(
            action_type="finalize",
            issue_type=TASKS[task_id].target_issue_type,
            priority=TASKS[task_id].target_priority,
            team=TASKS[task_id].target_team,
            severity=TASKS[task_id].target_severity,
            status=TASKS[task_id].target_status,
            tags=TASKS[task_id].required_tags,
            message=" ".join(TASKS[task_id].required_phrases),
            internal_note="validation run",
            confidence=0.9,
        )
        step_result = env.step(action)

        if not (0.0 <= step_result.reward.total <= 1.0):
            logger.error("Invalid reward for %s", task_id)
            raise SystemExit(1)

    logger.info("Validation SUCCESS: %s", json.dumps({"status": "ok", "tasks": TASK_ORDER}))


if __name__ == "__main__":
    main()
