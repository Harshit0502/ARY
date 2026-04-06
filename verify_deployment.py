import requests
import json
import os
from supportdesk_env.logging_config import get_logger

# Initialize professional logger
logger = get_logger("verify_deploy")

# Replace this with your actual Hugging Face Space URL from your browser
LIVE_URL = "https://Abhishek027-A-Chi.hf.space"

def verify_remote_deployment():
    """Talks to the live Hugging Face deployment to ensure it is healthy."""
    logger.info(f"--- Starting Deployment Verification for {LIVE_URL} ---")
    
    try:
        # 1. Health Check
        logger.info("🧪 Test 1: Checking System Health...")
        health_resp = requests.get(f"{LIVE_URL}/health", timeout=10)
        
        if health_resp.status_code == 200:
            logger.info(f"✅ SUCCESS: Server is Healthy. Tasks Available: {health_resp.json().get('task_count')}")
        else:
            logger.error(f"❌ FAILED: Status {health_resp.status_code} - {health_resp.text}")
            return

        # 2. Reset Check
        logger.info("\n🧪 Test 2: Starting a New Episode...")
        reset_resp = requests.post(f"{LIVE_URL}/reset", json={"task_id": "login_lockout"}, timeout=10)
        
        if reset_resp.status_code == 200:
            data = reset_resp.json()
            logger.info(f"✅ SUCCESS: Episode Started. Task: {data['observation']['title']}")
            logger.info(f"💡 Info: The live agent is ready to receive actions.")
        else:
            logger.error(f"❌ FAILED: Status {reset_resp.status_code}")

    except requests.exceptions.ConnectionError:
        logger.error("❌ FAILED: Could not connect to Hugging Face. Is the Space still 'Building'?")
    except Exception as e:
        logger.error(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    verify_remote_deployment()
