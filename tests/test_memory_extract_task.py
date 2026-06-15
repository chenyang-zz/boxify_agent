import pytest

from app.domain.models.long_term_memory import LongTermMemory, MemoryStatus
from app.domain.models.memory_graph import MemoryGraphStats
from app.infrastructure.external.task_dispatcher.celery_task_dispatcher import (
    CeleryTaskDispatcher,
)
from app.tasks.memory.extract import run_extract_memory
from tests.test_memory_service import InMemoryMemoryRepository, MemoryUnitOfWork


@pytest.mark.anyio
async def test_extract_memory_task_marks_memory_completed_with_graph_stats():
    repository = InMemoryMemoryRepository()
    memory = LongTermMemory(user_id="user-a", content="用户喜欢周杰伦。")
    await repository.save(memory)

    await run_extract_memory(
        memory_id=memory.id,
        uow_factory=lambda: MemoryUnitOfWork(repository),
        pipeline_factory=lambda user_id: FakePipeline(
            MemoryGraphStats(
                dialogue_id="dialogue-1",
                chunks=1,
                statements=1,
                entities=2,
                relations=1,
            )
        ),
    )

    updated = await repository.get_by_user("user-a", memory.id)
    assert updated.status == MemoryStatus.COMPLETED
    assert updated.graph_dialogue_id == "dialogue-1"
    assert updated.graph_stats == MemoryGraphStats(
        dialogue_id="dialogue-1",
        chunks=1,
        statements=1,
        entities=2,
        relations=1,
    )
    assert updated.error_msg is None


@pytest.mark.anyio
async def test_extract_memory_task_notifies_reflection_trigger_after_success():
    repository = InMemoryMemoryRepository()
    memory = LongTermMemory(user_id="user-a", content="用户喜欢周杰伦。")
    await repository.save(memory)
    dispatcher = FakeTaskDispatcher()

    await run_extract_memory(
        memory_id=memory.id,
        uow_factory=lambda: MemoryUnitOfWork(repository),
        pipeline_factory=lambda user_id: FakePipeline(
            MemoryGraphStats(
                dialogue_id="dialogue-1",
                chunks=1,
                statements=1,
                entities=11,
                relations=1,
            )
        ),
        task_dispatcher=dispatcher,
    )

    assert dispatcher.reflect_calls == [("user-a", 11)]


@pytest.mark.anyio
async def test_celery_dispatch_reflect_memory_waits_until_threshold(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(
        "app.infrastructure.external.task_dispatcher.celery_task_dispatcher.get_redis",
        lambda: FakeRedisClient(redis),
    )
    monkeypatch.setattr(
        "app.infrastructure.external.task_dispatcher.celery_task_dispatcher.get_settings",
        lambda: FakeSettings(threshold=10),
    )

    dispatcher = CeleryTaskDispatcher()
    assert await dispatcher.dispatch_reflect_memory("user-a", 3) is False
    assert await dispatcher.dispatch_reflect_memory("user-a", 6) is False

    assert redis.values == {"memory:reflection:pending:user-a": 9}
    assert redis.deleted == []


@pytest.mark.anyio
async def test_celery_dispatch_reflect_memory_dispatches_and_clears_at_threshold(
    monkeypatch,
):
    redis = FakeRedis()
    delayed = []
    monkeypatch.setattr(
        "app.infrastructure.external.task_dispatcher.celery_task_dispatcher.get_redis",
        lambda: FakeRedisClient(redis),
    )
    monkeypatch.setattr(
        "app.infrastructure.external.task_dispatcher.celery_task_dispatcher.get_settings",
        lambda: FakeSettings(threshold=10),
    )
    monkeypatch.setattr(
        "app.tasks.memory.reflect.reflect_memory_task.delay",
        lambda user_id: delayed.append(user_id),
    )

    dispatcher = CeleryTaskDispatcher()
    assert await dispatcher.dispatch_reflect_memory("user-a", 4) is False
    assert await dispatcher.dispatch_reflect_memory("user-a", 6) is True

    assert redis.values == {}
    assert redis.deleted == ["memory:reflection:pending:user-a"]
    assert delayed == ["user-a"]


@pytest.mark.anyio
async def test_celery_dispatch_reflect_memory_ignores_non_positive_count(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(
        "app.infrastructure.external.task_dispatcher.celery_task_dispatcher.get_redis",
        lambda: FakeRedisClient(redis),
    )

    assert await CeleryTaskDispatcher().dispatch_reflect_memory("user-a", 0) is False

    assert redis.values == {}
    assert redis.eval_calls == []


@pytest.mark.anyio
async def test_extract_memory_task_ignores_reflection_dispatch_failure():
    repository = InMemoryMemoryRepository()
    memory = LongTermMemory(user_id="user-a", content="用户喜欢周杰伦。")
    await repository.save(memory)

    await run_extract_memory(
        memory_id=memory.id,
        uow_factory=lambda: MemoryUnitOfWork(repository),
        pipeline_factory=lambda user_id: FakePipeline(
            MemoryGraphStats(
                dialogue_id="dialogue-1",
                chunks=1,
                statements=1,
                entities=11,
                relations=1,
            )
        ),
        task_dispatcher=ExplodingTaskDispatcher(),
    )

    updated = await repository.get_by_user("user-a", memory.id)
    assert updated.status == MemoryStatus.COMPLETED


@pytest.mark.anyio
async def test_extract_memory_task_marks_memory_failed_and_keeps_error_message():
    repository = InMemoryMemoryRepository()
    memory = LongTermMemory(user_id="user-a", content="用户喜欢周杰伦。")
    await repository.save(memory)

    await run_extract_memory(
        memory_id=memory.id,
        uow_factory=lambda: MemoryUnitOfWork(repository),
        pipeline_factory=lambda user_id: ExplodingPipeline(),
    )

    updated = await repository.get_by_user("user-a", memory.id)
    assert updated.status == MemoryStatus.FAILED
    assert "boom" in updated.error_msg


class FakePipeline:
    def __init__(self, stats):
        self.stats = stats

    async def extract_memory(self, memory_id: str, user_id: str, content: str):
        return self.stats


class ExplodingPipeline:
    async def extract_memory(self, memory_id: str, user_id: str, content: str):
        raise RuntimeError("boom")


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.deleted = []
        self.eval_calls = []

    async def eval(self, script, numkeys, key, amount, threshold):
        self.eval_calls.append((script, numkeys, key, amount, threshold))
        self.values[key] = self.values.get(key, 0) + int(amount)
        if self.values[key] >= int(threshold):
            await self.delete(key)
            return 1
        return 0

    async def delete(self, key):
        self.deleted.append(key)
        self.values.pop(key, None)


class FakeRedisClient:
    def __init__(self, redis):
        self.client = redis


class FakeSettings:
    def __init__(self, threshold):
        self.memory_reflection_trigger_threshold = threshold


class FakeTaskDispatcher:
    def __init__(self):
        self.reflect_calls = []

    async def dispatch_reflect_memory(self, user_id: str, entity_count: int) -> bool:
        self.reflect_calls.append((user_id, entity_count))
        return False


class ExplodingTaskDispatcher:
    async def dispatch_reflect_memory(self, user_id: str, entity_count: int) -> bool:
        raise RuntimeError("dispatch failed")
