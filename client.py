from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .models import SupportAction, StepResult, SupportObservation, SupportState


@dataclass
class SupportDeskClient:
    base_url: str = "http://127.0.0.1:7860"
    timeout: float = 30.0

    def reset(self, task_id: str | None = None, seed: int | None = None, episode_id: str | None = None) -> StepResult:
        payload = {"task_id": task_id, "seed": seed, "episode_id": episode_id}
        data = {k: v for k, v in payload.items() if v is not None}
        resp = requests.post(f"{self.base_url}/reset", json=data, timeout=self.timeout)
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def step(self, action: SupportAction) -> StepResult:
        resp = requests.post(f"{self.base_url}/step", json=action.model_dump(exclude_none=True), timeout=self.timeout)
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state(self) -> SupportState:
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return SupportState.model_validate(resp.json())

    def health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
