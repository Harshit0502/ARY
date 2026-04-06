from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..models import SupportAction, SupportObservation, SupportState, StepResult
from ..task_bank import TASK_ORDER, TASKS, public_tasks
from .environment import SupportDeskAppState

app = FastAPI(title="SupportDesk OpenEnv", version="0.1.0")
APP_STATE = SupportDeskAppState()


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = Field(default=None, description="Task to load. If omitted, the environment rotates.")
    seed: Optional[int] = Field(default=None, description="Optional seed for reproducibility.")
    episode_id: Optional[str] = Field(default=None, description="Optional episode identifier.")


@app.get("/health")
def health() -> Dict[str, Any]:
    return APP_STATE.health()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "supportdesk-openenv",
        "tasks": TASK_ORDER,
        "routes": ["/health", "/reset", "/step", "/state", "/tasks"],
    }


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return public_tasks()

@app.post("/reset", response_model=StepResult)
def reset(payload: ResetRequest) -> StepResult:
    if payload.task_id is not None and payload.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {payload.task_id}")
    return APP_STATE.env.reset(task_id=payload.task_id, seed=payload.seed, episode_id=payload.episode_id)


@app.post("/step", response_model=StepResult)
def step(action: SupportAction) -> StepResult:
    try:
        return APP_STATE.env.step(action)
    except Exception as exc:  # pragma: no cover - runtime protection
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=SupportState)
def state() -> SupportState:
    return APP_STATE.env.state


def main() -> None:
    import uvicorn
    uvicorn.run("supportdesk_env.server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":  # pragma: no cover
    main()