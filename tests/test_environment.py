from supportdesk_env.models import SupportAction
from supportdesk_env.server.environment import SupportDeskEnvironment
from supportdesk_env.task_bank import TASK_ORDER, TASKS, public_tasks


def test_three_tasks_exist() -> None:
    assert len(TASK_ORDER) >= 3


def test_reset_and_step_bounds() -> None:
    env = SupportDeskEnvironment()
    for task_id in TASK_ORDER:
        reset = env.reset(task_id=task_id)
        assert reset.observation.task_id == task_id

        action = SupportAction(
            action_type="finalize",
            issue_type=TASKS[task_id].target_issue_type,
            priority=TASKS[task_id].target_priority,
            team=TASKS[task_id].target_team,
            severity=TASKS[task_id].target_severity,
            status=TASKS[task_id].target_status,
            tags=TASKS[task_id].required_tags,
            message=" ".join(TASKS[task_id].required_phrases),
            internal_note="pytest",
            confidence=0.9,
        )

        result = env.step(action)
        assert 0.0 <= result.reward.total <= 1.0
        assert 0.0 <= result.observation.current_score <= 1.0
        assert result.done in {True, False}


def test_state_has_no_hidden_target() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id=TASK_ORDER[0])

    state_dict = env.state.model_dump()

    assert "hidden_target" not in state_dict


def test_public_task_manifest_is_redacted() -> None:
    manifest = public_tasks()

    assert isinstance(manifest, dict)
    assert len(manifest) >= 3

    forbidden_keys = {
        "target_team",
        "target_priority",
        "target_issue_type",
        "target_severity",
        "target_status",
        "required_phrases",
        "required_tags",
        "forbidden_phrases",
    }

    for task_id, task_data in manifest.items():
        assert task_id in TASK_ORDER
        assert isinstance(task_data, dict)

        leaked = forbidden_keys.intersection(task_data.keys())
        assert not leaked, f"Task {task_id} leaked private grader fields: {sorted(leaked)}"


def test_reset_observation_contains_only_public_fields() -> None:
    env = SupportDeskEnvironment()
    result = env.reset(task_id=TASK_ORDER[0])

    obs = result.observation.model_dump()

    forbidden_keys = {
        "target_team",
        "target_priority",
        "target_issue_type",
        "target_severity",
        "target_status",
        "required_phrases",
        "required_tags",
        "forbidden_phrases",
        "hidden_target",
    }

    leaked = forbidden_keys.intersection(obs.keys())
    assert not leaked, f"Observation leaked private fields: {sorted(leaked)}"

def test_state_still_tracks_runtime_fields() -> None:
    env = SupportDeskEnvironment()
    env.reset(task_id=TASK_ORDER[0])
    state_dict = env.state.model_dump()

    assert "episode_id" in state_dict
    assert "task_id" in state_dict
    assert "turn" in state_dict
    assert "best_score" in state_dict
    assert "transcript" in state_dict