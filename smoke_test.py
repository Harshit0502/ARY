import os
import requests
from supportdesk_env.logging_config import get_logger

# Initialize professional logger
logger = get_logger("smoke_test")

BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

def run_smoke_test():
    """Manually verifies the API connection and basic task flow."""
    try:
        # 1. Reset the environment
        logger.info("--- 1. Resetting Environment ---")
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "login_lockout"})
        
        if reset_resp.status_code != 200:
            logger.error(f"Reset failed (Status {reset_resp.status_code}): {reset_resp.text}")
            return

        data = reset_resp.json()
        logger.info(f"Connected to Task: {data['observation']['title']}")

        # 2. Submit a test action
        logger.info("--- 2. Submitting Manual Action ---")
        action = {
            "action_type": "classify",
            "team": "account_access",
            "priority": "medium",
            "issue_type": "login_lockout",
            "confidence": 0.9
        }
        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        
        if step_resp.status_code != 200:
            logger.error(f"Step failed (Status {step_resp.status_code}): {step_resp.text}")
            return

        result = step_resp.json()
        logger.info(f"Current Score: {result['observation']['current_score']}")
        logger.info(f"Feedback: {result['observation']['last_feedback'][:100]}...")
        logger.info(f"Done: {result['done']}")
        
    except Exception as e:
        logger.error(f"Failed to connect to simulation server: {e}")
        logger.info("Tip: Ensure 'uvicorn supportdesk_env.server.app:app' is running.")

if __name__ == "__main__":
    run_smoke_test()
