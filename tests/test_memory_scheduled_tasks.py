import pytest

from app.celery_app import build_memory_beat_schedule
from app.tasks.memory.scheduled import (
    run_scheduled_cluster_memory,
    run_scheduled_consolidate_memory,
    run_scheduled_reflect_memory,
)


@pytest.mark.anyio
async def test_scheduled_consolidate_dispatches_active_users_and_continues_after_failure():
    calls = []

    async def dispatch(user_id):
        calls.append(user_id)
        if user_id == "user-b":
            raise RuntimeError("dispatch failed")

    stats = await run_scheduled_consolidate_memory(
        uow_factory=lambda: FakeUnitOfWork(["user-a", "user-b", "user-c"]),
        dispatch=dispatch,
    )

    assert calls == ["user-a", "user-b", "user-c"]
    assert stats == {
        "users_scanned": 3,
        "dispatched": 2,
        "skipped": 0,
        "failed": 1,
    }


@pytest.mark.anyio
async def test_scheduled_cluster_dispatches_full_cluster_for_active_users():
    calls = []

    async def dispatch(user_id, dialogue_id=None):
        calls.append((user_id, dialogue_id))

    stats = await run_scheduled_cluster_memory(
        uow_factory=lambda: FakeUnitOfWork(["user-a", "user-b"]),
        dispatch=dispatch,
    )

    assert calls == [("user-a", None), ("user-b", None)]
    assert stats["users_scanned"] == 2
    assert stats["dispatched"] == 2
    assert stats["failed"] == 0


@pytest.mark.anyio
async def test_scheduled_reflect_dispatches_active_users():
    calls = []

    async def dispatch(user_id):
        calls.append(user_id)

    stats = await run_scheduled_reflect_memory(
        uow_factory=lambda: FakeUnitOfWork(["user-a"]),
        dispatch=dispatch,
    )

    assert calls == ["user-a"]
    assert stats == {
        "users_scanned": 1,
        "dispatched": 1,
        "skipped": 0,
        "failed": 0,
    }


def test_memory_beat_schedule_defaults_to_empty():
    schedule = build_memory_beat_schedule(
        FakeSettings(
            memory_maintenance_enabled=False,
            memory_scheduled_consolidate_enabled=True,
            memory_scheduled_cluster_enabled=True,
            memory_scheduled_reflect_enabled=True,
        )
    )

    assert schedule == {}


def test_memory_beat_schedule_registers_enabled_tasks():
    schedule = build_memory_beat_schedule(
        FakeSettings(
            memory_maintenance_enabled=True,
            memory_scheduled_consolidate_enabled=True,
            memory_scheduled_cluster_enabled=False,
            memory_scheduled_reflect_enabled=True,
            memory_maintenance_hour=4,
            memory_maintenance_minute=0,
        )
    )

    assert set(schedule) == {
        "memory-scheduled-consolidate",
        "memory-scheduled-reflect",
    }
    assert (
        schedule["memory-scheduled-consolidate"]["task"]
        == "app.tasks.memory_scheduled_consolidate"
    )
    assert (
        schedule["memory-scheduled-reflect"]["task"]
        == "app.tasks.memory_scheduled_reflect"
    )


class FakeSettings:
    def __init__(self, **overrides):
        self.memory_maintenance_enabled = False
        self.memory_scheduled_consolidate_enabled = False
        self.memory_scheduled_cluster_enabled = False
        self.memory_scheduled_reflect_enabled = False
        self.memory_maintenance_hour = 4
        self.memory_maintenance_minute = 0
        for key, value in overrides.items():
            setattr(self, key, value)


class FakeUnitOfWork:
    def __init__(self, active_user_ids):
        self.user = FakeUserRepository(active_user_ids)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class FakeUserRepository:
    def __init__(self, active_user_ids):
        self.active_user_ids = active_user_ids

    async def list_active_ids(self):
        return self.active_user_ids
