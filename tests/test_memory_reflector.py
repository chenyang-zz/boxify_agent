import pytest

from app.domain.models.memory_graph import EntityNode, MemoryReflectStats
from app.domain.services.memory.insight_generator import ReflectedInsight
from app.domain.services.memory.reflector import MemoryReflector


@pytest.mark.anyio
async def test_memory_reflector_skips_when_entities_are_insufficient():
    repository = FakeReflectionGraphRepository(top_entities=[])
    reflector = MemoryReflector(
        user_id="user-a",
        graph_repository=repository,
        insight_generator=FakeInsightGenerator([]),
        embedding=FakeEmbedding(),
    )

    stats = await reflector.reflect()

    assert stats == MemoryReflectStats(insights=0, skipped="too_few_entities")
    assert repository.insight_writes == []


@pytest.mark.anyio
async def test_memory_reflector_upserts_insights_with_entity_sources():
    repository = FakeReflectionGraphRepository()
    reflector = MemoryReflector(
        user_id="user-a",
        graph_repository=repository,
        insight_generator=FakeInsightGenerator(
            [
                ReflectedInsight(
                    theme="音乐偏好",
                    content="用户偏好华语流行音乐。",
                    based_on=["周杰伦", "林俊杰"],
                    importance=0.8,
                    confidence=0.9,
                )
            ]
        ),
        embedding=FakeEmbedding(),
    )

    stats = await reflector.reflect()

    assert stats == MemoryReflectStats(insights=1)
    assert repository.insight_writes == [
        {
            "user_id": "user-a",
            "theme": "音乐偏好",
            "content": "用户偏好华语流行音乐。",
            "embedding": [0.5],
            "importance": 0.8,
            "confidence": 0.9,
            "source_count": 2,
            "entity_ids": ["entity-1", "entity-2"],
        }
    ]


@pytest.mark.anyio
async def test_memory_reflector_skips_single_bad_insight_write():
    repository = FakeReflectionGraphRepository(fail_themes={"音乐偏好"})
    reflector = MemoryReflector(
        user_id="user-a",
        graph_repository=repository,
        insight_generator=FakeInsightGenerator(
            [
                ReflectedInsight(
                    theme="音乐偏好",
                    content="用户偏好华语流行音乐。",
                    based_on=["周杰伦"],
                ),
                ReflectedInsight(
                    theme="工作偏好",
                    content="用户偏好安静工作环境。",
                    based_on=["用户"],
                ),
            ]
        ),
        embedding=FakeEmbedding(),
    )

    stats = await reflector.reflect()

    assert stats == MemoryReflectStats(insights=1)
    assert [write["theme"] for write in repository.insight_writes] == ["工作偏好"]


class FakeReflectionGraphRepository:
    def __init__(self, top_entities=None, fail_themes=None):
        self.top_entities = (
            [
                EntityNode(
                    id="entity-1",
                    user_id="user-a",
                    name="周杰伦",
                    type="生命体",
                    description="歌手",
                    core_facts=["用户长期喜欢周杰伦"],
                    traits=["偏好华语流行"],
                ),
                EntityNode(
                    id="entity-2",
                    user_id="user-a",
                    name="林俊杰",
                    type="生命体",
                    description="歌手",
                ),
                EntityNode(id="entity-3", user_id="user-a", name="用户", type="生命体"),
                EntityNode(id="entity-4", user_id="user-a", name="上海", type="地点"),
                EntityNode(id="entity-5", user_id="user-a", name="安静环境", type="概念"),
            ]
            if top_entities is None
            else top_entities
        )
        self.fail_themes = fail_themes or set()
        self.insight_writes = []

    async def reflection_top_entities(self, user_id, top_k):
        assert user_id == "user-a"
        assert top_k == 30
        return self.top_entities

    async def reflection_entity_statements(self, user_id, entity_id, limit):
        assert user_id == "user-a"
        assert limit == 5
        return [f"{entity_id} statement 1", f"{entity_id} statement 2"]

    async def upsert_insight(
        self,
        user_id,
        theme,
        content,
        embedding,
        importance,
        confidence,
        source_count,
        entity_ids,
    ):
        if theme in self.fail_themes:
            raise RuntimeError("write failed")
        self.insight_writes.append(
            {
                "user_id": user_id,
                "theme": theme,
                "content": content,
                "embedding": embedding,
                "importance": importance,
                "confidence": confidence,
                "source_count": source_count,
                "entity_ids": entity_ids,
            }
        )


class FakeInsightGenerator:
    def __init__(self, insights):
        self.insights = insights
        self.calls = []

    async def generate(self, memory_block, min_insights, max_insights):
        assert "记忆清单" not in memory_block
        assert min_insights == 1
        assert max_insights == 5
        self.calls.append(memory_block)
        return self.insights


class FakeEmbedding:
    async def embed(self, texts):
        return [[0.5] for _ in texts]

    async def embed_one(self, text):
        return [0.5]

    @property
    def model_name(self):
        return "fake-embedding"
