import pytest

from app.domain.models.memory_graph import MemoryCommunityClusterStats
from app.tasks.memory.cluster import run_cluster_memory


@pytest.mark.anyio
async def test_cluster_memory_task_returns_clusterer_stats():
    stats = await run_cluster_memory(
        user_id="user-a",
        dialogue_id="dialogue-1",
        clusterer_factory=lambda user_id: FakeClusterer.create(user_id),
    )

    assert stats == MemoryCommunityClusterStats(
        communities=2,
        assigned_entities=3,
        merged_communities=1,
        enhanced_communities=2,
    )
    assert FakeClusterer.calls == [("user-a", "dialogue-1")]


class FakeClusterer:
    calls = []

    @classmethod
    async def create(cls, user_id):
        return cls(user_id)

    def __init__(self, user_id):
        self.user_id = user_id

    async def cluster(self, dialogue_id=None):
        FakeClusterer.calls.append((self.user_id, dialogue_id))
        return MemoryCommunityClusterStats(
            communities=2,
            assigned_entities=3,
            merged_communities=1,
            enhanced_communities=2,
        )
