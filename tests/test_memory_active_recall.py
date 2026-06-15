import pytest

from app.domain.models.memory_graph import (
    GraphRelationFact,
    InsightResult,
    MemoryGraphResult,
)
from app.domain.services.memory.active_recall import MemoryActiveRecall


@pytest.mark.anyio
async def test_memory_active_recall_formats_insights_and_entity_facts():
    recall = MemoryActiveRecall(
        user_id="user-a",
        graph_repository=FakeRecallGraphRepository(),
        embedding=FakeEmbedding(),
    )

    context = await recall.recall_context("我最近想听什么歌？")

    assert "关于用户的已知信息" in context
    assert "用户偏好华语流行音乐" in context
    assert "周杰伦：歌手" in context
    assert "用户 偏好 周杰伦" in context
    assert FakeRecallGraphRepository.last_query_embedding == [0.7]


@pytest.mark.anyio
async def test_memory_active_recall_returns_empty_on_embedding_failure():
    recall = MemoryActiveRecall(
        user_id="user-a",
        graph_repository=FakeRecallGraphRepository(),
        embedding=ExplodingEmbedding(),
    )

    assert await recall.recall_context("周杰伦") == ""


@pytest.mark.anyio
async def test_memory_active_recall_returns_empty_on_graph_failure():
    recall = MemoryActiveRecall(
        user_id="user-a",
        graph_repository=ExplodingRecallGraphRepository(),
        embedding=FakeEmbedding(),
    )

    assert await recall.recall_context("周杰伦") == ""


class FakeRecallGraphRepository:
    last_query_embedding = None

    async def search_insights_by_vector(self, user_id, query_embedding, top_k):
        assert user_id == "user-a"
        assert top_k == 3
        FakeRecallGraphRepository.last_query_embedding = query_embedding
        return [
            InsightResult(
                id="insight-1",
                theme="音乐偏好",
                content="用户偏好华语流行音乐。",
                score=0.9,
            )
        ]

    async def search(self, user_id, query, top_k, query_embedding=None):
        assert user_id == "user-a"
        assert query == "我最近想听什么歌？"
        assert top_k == 3
        assert query_embedding == [0.7]
        return [
            MemoryGraphResult(
                entity_id="entity-1",
                entity_name="周杰伦",
                entity_type="生命体",
                description="歌手",
                score=0.91,
                relations=[
                    GraphRelationFact(
                        name="偏好",
                        direction="incoming",
                        neighbor_name="用户",
                        neighbor_type="生命体",
                        evidence="用户喜欢周杰伦。",
                    )
                ],
            )
        ]


class ExplodingRecallGraphRepository:
    async def search_insights_by_vector(self, user_id, query_embedding, top_k):
        raise RuntimeError("neo4j unavailable")

    async def search(self, user_id, query, top_k, query_embedding=None):
        raise RuntimeError("neo4j unavailable")


class FakeEmbedding:
    async def embed(self, texts):
        return [[0.7] for _ in texts]

    async def embed_one(self, text):
        return [0.7]

    @property
    def model_name(self):
        return "fake-embedding"


class ExplodingEmbedding(FakeEmbedding):
    async def embed_one(self, text):
        raise RuntimeError("embedding unavailable")
