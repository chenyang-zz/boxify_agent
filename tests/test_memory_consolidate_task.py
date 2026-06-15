import pytest

from app.domain.models.memory_graph import MemoryConsolidationStats
from app.tasks.memory.consolidate import run_consolidate_memory


@pytest.mark.anyio
async def test_consolidate_memory_task_returns_service_stats():
    stats = await run_consolidate_memory(
        user_id="user-a",
        service_factory=lambda user_id: FakeConsolidationService.create(user_id),
    )

    assert stats == MemoryConsolidationStats(
        promoted_entities=2,
        promoted_statements=3,
        enhanced_profiles=1,
    )


class FakeConsolidationService:
    def __init__(self, user_id):
        self.user_id = user_id

    @classmethod
    async def create(cls, user_id):
        return cls(user_id)

    async def consolidate(self):
        assert self.user_id == "user-a"
        return MemoryConsolidationStats(
            promoted_entities=2,
            promoted_statements=3,
            enhanced_profiles=1,
        )
