import pytest

from app.domain.models.memory_graph import (
    GraphRelationFact,
    InsightResult,
    MemoryActiveRecallCommunityResult,
    MemoryActiveRecallEventResult,
    MemoryGraphResult,
    MemoryTimelineParticipantResult,
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
    assert "相关主题社区：音乐偏好：用户经常提到华语流行与演唱会安排。" in context
    assert "相关经历：2026-06-01T20:00:00 周杰伦演唱会 - 用户、周杰伦" in context
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
async def test_memory_active_recall_skips_failed_lanes():
    recall = MemoryActiveRecall(
        user_id="user-a",
        graph_repository=PartiallyExplodingRecallGraphRepository(),
        embedding=FakeEmbedding(),
    )

    context = await recall.recall_context("我最近想听什么歌？")

    assert "用户偏好华语流行音乐" in context
    assert "周杰伦" not in context
    assert "相关主题社区" not in context
    assert "相关经历" not in context


@pytest.mark.anyio
async def test_memory_active_recall_honors_community_and_event_switches(monkeypatch):
    settings = FakeRecallSettings(
        memory_active_recall_include_communities=False,
        memory_active_recall_include_events=False,
    )
    monkeypatch.setattr(
        "app.domain.services.memory.active_recall.get_settings",
        lambda: settings,
    )
    repository = FakeRecallGraphRepository()
    recall = MemoryActiveRecall(
        user_id="user-a",
        graph_repository=repository,
        embedding=FakeEmbedding(),
    )

    context = await recall.recall_context("我最近想听什么歌？")

    assert "用户偏好华语流行音乐" in context
    assert repository.community_calls == 0
    assert repository.event_calls == 0


class FakeRecallGraphRepository:
    last_query_embedding = None

    def __init__(self):
        self.community_calls = 0
        self.event_calls = 0

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

    async def search_communities_by_vector(self, user_id, query_embedding, top_k):
        assert user_id == "user-a"
        assert query_embedding == [0.7]
        assert top_k == 2
        self.community_calls += 1
        return [
            MemoryActiveRecallCommunityResult(
                id="community-1",
                name="音乐偏好",
                summary="用户经常提到华语流行与演唱会安排。",
                score=0.88,
            )
        ]

    async def search_events_by_vector_or_text(
        self, user_id, query, query_embedding, top_k
    ):
        assert user_id == "user-a"
        assert query_embedding == [0.7]
        assert top_k == 2
        self.event_calls += 1
        return [
            MemoryActiveRecallEventResult(
                id="event-1",
                title="周杰伦演唱会",
                event_time="2026-06-01T20:00:00",
                participants=[
                    MemoryTimelineParticipantResult(
                        entity_id="entity-user",
                        name="用户",
                        type="生命体",
                    ),
                    MemoryTimelineParticipantResult(
                        entity_id="entity-jay",
                        name="周杰伦",
                        type="生命体",
                    ),
                ],
                score=0.89,
            )
        ]


class PartiallyExplodingRecallGraphRepository(FakeRecallGraphRepository):
    async def search_insights_by_vector(self, user_id, query_embedding, top_k):
        return [
            InsightResult(
                id="insight-1",
                theme="音乐偏好",
                content="用户偏好华语流行音乐。",
                score=0.9,
            )
        ]

    async def search(self, user_id, query, top_k, query_embedding=None):
        raise RuntimeError("neo4j unavailable")

    async def search_communities_by_vector(self, user_id, query_embedding, top_k):
        raise RuntimeError("community search unavailable")

    async def search_events_by_vector_or_text(
        self, user_id, query, query_embedding, top_k
    ):
        raise RuntimeError("event search unavailable")


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


class FakeRecallSettings:
    def __init__(self, **overrides):
        self.memory_active_recall_timeout_seconds = 3.5
        self.memory_active_recall_entity_top_k = 3
        self.memory_active_recall_insight_top_k = 3
        self.memory_active_recall_min_score = 0.72
        self.memory_active_recall_max_chars = 1200
        self.memory_active_recall_community_top_k = 2
        self.memory_active_recall_event_top_k = 2
        self.memory_active_recall_include_communities = True
        self.memory_active_recall_include_events = True
        for key, value in overrides.items():
            setattr(self, key, value)
