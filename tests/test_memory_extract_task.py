import pytest

from app.domain.models.long_term_memory import LongTermMemory, MemoryStatus
from app.domain.models.memory_graph import MemoryGraphStats
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
