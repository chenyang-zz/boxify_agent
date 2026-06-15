import pytest

from app.domain.models.memory_graph import MemoryReflectStats
from app.tasks.memory.reflect import run_reflect_memory


@pytest.mark.anyio
async def test_reflect_memory_task_returns_reflector_stats():
    stats = await run_reflect_memory(
        user_id="user-a",
        reflector_factory=lambda user_id: FakeReflector.create(user_id),
    )

    assert stats == MemoryReflectStats(insights=2)


class FakeReflector:
    def __init__(self, user_id):
        self.user_id = user_id

    @classmethod
    async def create(cls, user_id):
        return cls(user_id)

    async def reflect(self):
        assert self.user_id == "user-a"
        return MemoryReflectStats(insights=2)
