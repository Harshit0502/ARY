# Getting Started with SupportDesk OpenEnv

Welcome to the SupportDesk OpenEnv project! This guide will take you from a fresh clone to a fully running AI agent evaluation system in under 5 minutes.

---

## 📋 Prerequisites
- Python 3.11 or higher
- (Optional) Docker

---

## 🛠️ Step 1: Environment Setup

First, create a virtual environment and install the package in "editable" mode. This ensures that any changes you make to the source code are applied immediately.

```powershell
# Create the virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -e .
pip install python-dotenv
```

---

## 🔑 Step 2: Configuration

The system uses environment variables for API keys and model settings. Create a file named `.env` in the root directory:

```text
# .env file
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
HF_TOKEN=your_hugging_face_token_here
```

---

## 🖥️ Step 3: Running the Simulation Server

The simulation environment is a FastAPI web server. You must have this running before the AI agent can play the scenarios.

```powershell
uvicorn supportdesk_env.server.app:app --host 127.0.0.1 --port 7860
```
*Note: Keep this terminal window open. If you see "Application startup complete," the server is ready.*

---

## 🤖 Step 4: Running the AI Agent

Open a **separate terminal window** to run the agent. The agent will read the scenarios from the server and attempt to solve them using your AI model.

### Complete Run (All Tasks)
```powershell
python inference.py
```

### Run a Single Task
```powershell
python inference.py --tasks login_lockout
```

### Dry Run (Offline Mode)
Test the whole system without using any AI credits:
```powershell
python inference.py --offline
```

---

## 🧪 Step 5: Verification & Testing

### Automated Unit Tests
Run the internal logic tests (checks the grader and environment mechanics):
```powershell
pytest
```

### Manual API check
Check if the server is healthy using `curl`:
```powershell
curl http://127.0.0.1:7860/health
```

### Smoke Test Script
We have included a specific script to verify the end-to-end API connection:
```powershell
python smoke_test.py
```

---

## 🐳 Step 6: Docker Deployment (Optional)

If you want to run the environment inside a container:

```bash
# Build the image
docker build -t supportdesk-openenv .

# Run the container
docker run -p 7860:7860 supportdesk-openenv
```

---

## 📁 Project Structure Quick-Map
- **`supportdesk_env/`**: The core simulation logic.
- **`task_bank.py`**: Where the customer scenarios are defined.
- **`inference.py`**: The main entry point for the AI agent.
- **`app.py`**: The FastAPI server entry point.
- **`grader.py`**: Logic that scores the AI's performance.
