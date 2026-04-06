# SupportDesk Environment Breakdown

This document provides a comprehensive, step-by-step breakdown of how the SupportDesk Environment repository works, what each file does, and how the entire system functions from end-to-end.

## 1. The High-Level Concept
This repository is an **"OpenEnv" style simulation**, similar to a flight simulator, but designed for an AI customer support agent. 

When evaluating an AI for customer support, it's risky to test it on real unhappy customers immediately. Instead, it's placed in this simulation:
1. It simulates angry customers creating tickets.
2. It accepts actions taken by an AI agent (e.g., "route to billing" or "send a refund").
3. It grades those actions probabilistically based on predefined rules, determining if the AI followed company policy correctly.

---

## 2. File-by-File Breakdown

### A. Core Configuration
*   **`Dockerfile`**: A blueprint for creating a standardized running environment. It bundles the code and dependencies (like FastAPI) into a container, ensuring the application runs identically on any computer or cloud server.
*   **`.env`**: Securely stores secret keys like `HF_TOKEN` and connection URLs, keeping them out of the source code.
*   **`pyproject.toml`**: The modern Python configuration file. It lists external packages required by the project (e.g., `fastapi`, `requests`, `pydantic`).

### B. The Rules of the Simulation (`supportdesk_env/`)
*   **`task_bank.py`**: The central repository for scenarios. It defines three distinct customer issues:
    1.  *Easy*: A customer locked out after a password reset.
    2.  *Medium*: A customer charged twice requesting a refund.
    3.  *Hard*: A massive European server outage requiring coordination. 
    It also strictly defines "correct" actions (e.g., "The target team must be *billing* and you must mention *3-5 business days*").
*   **`models.py`**: Defines data structures using Pydantic. It dictates exactly what fields an AI agent can include in an action (`team`, `priority`, `message`, `refund_amount`) and what information the AI will receive as an observation.
*   **`grader.py`**: The evaluation engine. When the AI takes an action, this script compares it against the rules in `task_bank.py`. It deducts points if the AI hallucinated, used forbidden phrases, or routed the ticket incorrectly, generating a score from 0.0 to 1.0.

### C. The Simulation Server (`supportdesk_env/server/`)
*   **`environment.py`**: Functions as the game engine. It tracks the "state" (current turn, AI's score). When a new task begins, it selects a scenario from the `task_bank`. When the AI executes a step, it passes the action to the `grader`, records the score, and advances the turn.
*   **`app.py`**: The web interface. It utilizes FastAPI to expose the game engine (`environment.py`) over HTTP, allowing external AI scripts to interact with the simulation via requests (`/reset`, `/step`, `/health`).

### D. The AI Player
*   **`client.py`**: A helper utility. Instead of making raw HTTP requests, the AI player script can use this Python object to easily communicate with the server (e.g., `client.reset()`, `client.step()`).
*   **`inference.py`**: The primary "Agent Player" testing script. It serves as an example of an AI attempting to navigate the simulation scenarios.

---

## 3. Step-by-Step Flow: What Happens When You Run It?

When you execute `python inference.py`, the following chronological sequence of events occurs simultaneously on the server and the client:

### Step 1: Initialization
1.  In `inference.py`, the script loads the `.env` file credentials.
2.  It initializes an `OpenAI` client (configured to communicate with Hugging Face).
3.  It initializes the `SupportDeskClient`, enabling it to communicate with the locally running `uvicorn` server instance of `app.py`.

### Step 2: Starting a Scenario (The Task)
1.  The `inference` script asks the server to start a scenario using `env.reset(task_id="login_lockout")`.
2.  The server (`environment.py`) loads the "Easy" ticket from `task_bank.py`.
3.  The server replies with a `SupportObservation` containing the simulated customer's email summary.

### Step 3: The AI "Thinks"
1.  In `inference.py`, the function `_call_model_action` packages the rules and the customer's email into a comprehensive prompt.
2.  It sends this prompt to the AI Model hosted on Hugging Face using its API key.
3.  The model generates a JSON object representing a recommended action (e.g., `{"action_type": "finalize", "team": "account_access", "message": "Try checking your spam..."}`).

### Step 4: Grading the AI (The Step)
1.  The client sends the generated JSON action back to the server using `env.step(action)`.
2.  The server receives it and passes it to `grader.py`.
3.  `grader.py` compares the AI's JSON against the strict expected answers in `task_bank.py`, checking for proper routing and forbidden terms.
4.  It assigns a score and generates feedback (e.g., "Good progress, but you forgot to set the severity.").

### Step 5: Iteration and Result
1.  The server updates its internal state and returns the score back to the `inference` script.
2.  The loop terminates if the score is perfect (1.0) or if the AI exhausts its allotted attempts (4 turns).
3.  The script proceeds to the next task (Medium, then Hard), eventually printing the Mean Score across all simulated tasks.

---

## Summary Statement
This codebase is an automated evaluation environment for customer support AI agents. It consists of a FastAPI server holding three standard tricky scenarios. It evaluates agents based on how well they route tickets, adhere to tone constraints, and follow strict operational policies, grading them instantly using a deterministic rubric. The `inference.py` script acts as the test runner, plugging a real LLM into the local server to see how well it scores against our constraints.
