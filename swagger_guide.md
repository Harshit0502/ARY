# Swagger API Testing Guide: SupportDesk OpenEnv

Your project is built with **FastAPI**, which means it has "out-of-the-box" interactive documentation called Swagger. This is the best way to demonstrate the API to your superiors.

---

## 🚀 How to Launch
1. Start your server:
   ```powershell
   uvicorn supportdesk_env.server.app:app --host 127.0.0.1 --port 7860
   ```
2. Open your browser and navigate to:
   **[http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs)**

---

## 🧪 Step-by-Step Testing Instructions

### 1. Start a New Episode (`POST /reset`)
This endpoint initializes a new support scenario.
1. Click on the **`POST /reset`** bar.
2. Click **"Try it out"**.
3. Use the following JSON as a payload:
   ```json
   {
     "task_id": "login_lockout"
   }
   ```
4. Click **"Execute"**. You should see the `inbox_summary` and a `turn` of `0`.

---

### 2. Take an Action (`POST /step`)
This is where you "play the game" as the agent.
1. Click on the **`POST /step`** bar.
2. Click **"Try it out"**.
3. Use this JSON payload (this simulates a high-scoring action for the Login scenario):
   ```json
   {
     "action_type": "finalize",
     "issue_type": "account_access",
     "priority": "medium",
     "team": "account_access",
     "status": "responded",
     "tags": ["login", "password-reset"],
     "message": "Sorry for the trouble. Please check your spam folder for the reset link and try the 2FA flow again.",
     "internal_note": "Successfully triaged login lockout.",
     "confidence": 1.0
   }
   ```
4. Click **"Execute"**. You should see a `reward` of `1.0` if you matched all the criteria!

---

### 3. Inspect Current State (`GET /state`)
Use this if you want to see the "hidden" progress of the simulation.
1. Click on **`GET /state`**.
2. Click **"Try it out"** -> **"Execute"**.
3. It will show you the `episode_id`, the current `best_score`, and the full `transcript` of all actions taken so far.

---

### 4. System Health (`GET /health`)
A quick check to make sure the environment is configured correctly.
1. Click on **`GET /health`** -> **"Execute"**.
2. **Expected Response**: `{"status":"healthy","task_count":3}`

---

## 🛠️ Optional: Improving Swagger UI
Currently, you have to copy-paste the JSON examples above. We can update **`supportdesk_env/models.py`** to include these examples as "defaults" in Swagger. 

If you want me to do this, let me know! It makes the API look much more professional for non-developers.
