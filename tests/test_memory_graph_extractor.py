import pytest

from app.domain.models.memory_graph import EntityNode, MemoryGraphStats, StatementNode
from app.domain.services.memory.fact_extractor import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedTriplet,
    ExtractedTriplets,
)
from app.domain.services.memory.graph_extractor import MemoryGraphExtractor


@pytest.mark.anyio
async def test_memory_graph_extractor_writes_four_layer_graph_with_deduped_entities():
    fact_extractor = FakeFactExtractor(
        statements=[
            StatementNode(
                id="statement-1",
                user_id="user-a",
                chunk_id="chunk-1",
                index=0,
                text="用户住在上海。",
                statement_type="FACT",
                temporal_type="STATIC",
                importance=0.7,
                confidence=0.9,
                valid_at="2026-06-16T09:00:00",
            ),
            StatementNode(
                id="statement-2",
                user_id="user-a",
                chunk_id="chunk-1",
                index=1,
                text="用户喜欢周杰伦。",
                statement_type="FACT",
                temporal_type="STATIC",
                importance=0.8,
                confidence=0.95,
            ),
        ],
        triplets=ExtractedTriplets(
            entities=[
                ExtractedEntity(
                    entity_idx=1,
                    name="用户",
                    type="生命体",
                    description="当前用户",
                ),
                ExtractedEntity(
                    entity_idx=2,
                    name="上海",
                    type="地点设施",
                    description="城市",
                ),
                ExtractedEntity(
                    entity_idx=3,
                    name="用户",
                    type="生命体",
                    description="当前用户",
                ),
                ExtractedEntity(
                    entity_idx=4,
                    name="周杰伦",
                    type="Person",
                    description="歌手",
                ),
            ],
            triplets=[
                ExtractedTriplet(
                    subject_id=1,
                    predicate="位于",
                    object_id=2,
                    evidence="用户住在上海。",
                    valid_at="2026-06-16T09:00:00",
                ),
                ExtractedTriplet(
                    subject_id=3,
                    predicate="LIKES",
                    object_id=4,
                    evidence="用户喜欢周杰伦。",
                    invalid_at="2026-07-01T00:00:00",
                ),
            ],
            events=[
                ExtractedEvent(
                    title="参加周杰伦演唱会",
                    description="用户昨天参加了周杰伦演唱会",
                    event_time="2026-06-15T20:00:00",
                    participants=["用户", "周杰伦", "不存在的实体"],
                ),
                ExtractedEvent(
                    title="",
                    description="空标题事件应跳过",
                    event_time="NULL",
                    participants=["用户"],
                ),
            ],
        ),
    )
    repository = FakeGraphRepository()
    extractor = MemoryGraphExtractor(
        fact_extractor=fact_extractor,
        embedding=FakeEmbedding(),
        graph_repository=repository,
    )

    stats = await extractor.extract_memory(
        memory_id="mem-1",
        user_id="user-a",
        content="用户住在上海。用户喜欢周杰伦。",
    )

    graph = repository.saved_graphs[0]
    assert fact_extractor.chunk_texts == [["用户住在上海。用户喜欢周杰伦。"]]
    assert graph.dialogue.memory_id == "mem-1"
    assert graph.dialogue.user_id == "user-a"
    assert len(graph.chunks) == 1
    assert [statement.text for statement in graph.statements] == [
        "用户住在上海。",
        "用户喜欢周杰伦。",
    ]
    assert {(entity.name, entity.type) for entity in graph.entities} == {
        ("用户", "生命体"),
        ("上海", "地点设施"),
        ("周杰伦", "其他"),
    }
    assert all(entity.embedding for entity in graph.entities)
    assert [(rel.name, rel.evidence) for rel in graph.relations] == [
        ("位于", "用户住在上海。"),
        ("关联于", "用户喜欢周杰伦。"),
    ]
    assert graph.statements[0].valid_at.isoformat() == "2026-06-16T09:00:00"
    assert graph.relations[0].valid_at.isoformat() == "2026-06-16T09:00:00"
    assert graph.relations[0].invalid_at is None
    assert graph.relations[1].valid_at is None
    assert graph.relations[1].invalid_at.isoformat() == "2026-07-01T00:00:00"
    assert graph.relations[0].source_entity_id == graph.relations[1].source_entity_id
    assert len(graph.events) == 1
    assert graph.events[0].title == "参加周杰伦演唱会"
    assert graph.events[0].event_time.isoformat() == "2026-06-15T20:00:00"
    assert {(edge.event_id, edge.entity_id) for edge in graph.involves} == {
        (graph.events[0].id, graph.relations[0].source_entity_id),
        (graph.events[0].id, graph.relations[1].target_entity_id),
    }
    assert stats == MemoryGraphStats(
        dialogue_id=graph.dialogue.id,
        chunks=1,
        statements=2,
        entities=3,
        relations=2,
        events=1,
        involves=2,
    )


@pytest.mark.anyio
async def test_memory_graph_extractor_handles_empty_fact_result():
    repository = FakeGraphRepository()
    extractor = MemoryGraphExtractor(
        fact_extractor=FakeFactExtractor(
            statements=[],
            triplets=ExtractedTriplets(),
        ),
        embedding=FakeEmbedding(),
        graph_repository=repository,
    )

    stats = await extractor.extract_memory(
        memory_id="mem-1",
        user_id="user-a",
        content="用户住在上海。",
    )

    graph = repository.saved_graphs[0]
    assert graph.statements == []
    assert graph.entities == []
    assert graph.relations == []
    assert stats.dialogue_id == graph.dialogue.id


@pytest.mark.anyio
async def test_memory_graph_extractor_reuses_existing_same_name_entity_by_type():
    fact_extractor = FakeFactExtractor(
        statements=[
            StatementNode(
                id="statement-1",
                user_id="user-a",
                chunk_id="chunk-1",
                index=0,
                text="用户喜欢周杰伦。",
            )
        ],
        triplets=ExtractedTriplets(
            entities=[
                ExtractedEntity(
                    entity_idx=1,
                    name="用户",
                    type="生命体",
                    description="当前用户",
                ),
                ExtractedEntity(
                    entity_idx=2,
                    name="周杰伦",
                    type="生命体",
                    description="歌手",
                ),
            ],
            triplets=[
                ExtractedTriplet(
                    subject_id=1,
                    predicate="偏好",
                    object_id=2,
                    evidence="用户喜欢周杰伦。",
                )
            ],
        ),
    )
    repository = FakeGraphRepository()
    repository.existing_by_type = {
        "生命体": [
            EntityNode(
                id="existing-user-entity",
                user_id="user-a",
                name="用户",
                type="生命体",
                description="历史用户节点",
            )
        ]
    }
    extractor = MemoryGraphExtractor(
        fact_extractor=fact_extractor,
        embedding=FakeEmbedding(),
        graph_repository=repository,
    )

    await extractor.extract_memory(
        memory_id="mem-2",
        user_id="user-a",
        content="用户喜欢周杰伦。",
    )

    graph = repository.saved_graphs[0]
    user_entity = next(entity for entity in graph.entities if entity.name == "用户")
    assert user_entity.id == "existing-user-entity"
    assert graph.relations[0].source_entity_id == "existing-user-entity"
    assert graph.mentions[0].entity_id == "existing-user-entity"


@pytest.mark.anyio
async def test_memory_graph_extractor_redirects_event_participants_to_existing_entities():
    fact_extractor = FakeFactExtractor(
        statements=[
            StatementNode(
                id="statement-1",
                user_id="user-a",
                chunk_id="chunk-1",
                index=0,
                text="用户昨天参加了周杰伦演唱会。",
            )
        ],
        triplets=ExtractedTriplets(
            entities=[
                ExtractedEntity(
                    entity_idx=1,
                    name="用户",
                    type="生命体",
                    description="当前用户",
                ),
                ExtractedEntity(
                    entity_idx=2,
                    name="周杰伦",
                    type="生命体",
                    description="歌手",
                ),
            ],
            triplets=[
                ExtractedTriplet(
                    subject_id=1,
                    predicate="关联于",
                    object_id=2,
                    evidence="用户昨天参加了周杰伦演唱会。",
                )
            ],
            events=[
                ExtractedEvent(
                    title="参加周杰伦演唱会",
                    description="用户昨天参加了周杰伦演唱会",
                    event_time="NULL",
                    participants=["用户", "周杰伦"],
                )
            ],
        ),
    )
    repository = FakeGraphRepository()
    repository.existing_by_type = {
        "生命体": [
            EntityNode(
                id="existing-user-entity",
                user_id="user-a",
                name="用户",
                type="生命体",
                description="历史用户节点",
            )
        ]
    }
    extractor = MemoryGraphExtractor(
        fact_extractor=fact_extractor,
        embedding=FakeEmbedding(),
        graph_repository=repository,
    )

    stats = await extractor.extract_memory(
        memory_id="mem-3",
        user_id="user-a",
        content="用户昨天参加了周杰伦演唱会。",
    )

    graph = repository.saved_graphs[0]
    assert stats.events == 1
    assert "existing-user-entity" in {edge.entity_id for edge in graph.involves}


class FakeFactExtractor:
    def __init__(
        self, statements: list[StatementNode], triplets: ExtractedTriplets
    ) -> None:
        self._statements = statements
        self._triplets = triplets
        self.chunk_texts = []

    async def extract_statements(self, chunks, dialog_at=None):
        self.chunk_texts.append([chunk.text for chunk in chunks])
        return self._statements

    async def extract_triplets(self, statements, dialog_at=None):
        return self._triplets


class FakeEmbedding:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, texts):
        return [[float(index + 1)] * 4 for index, _ in enumerate(texts)]

    async def embed_one(self, text):
        return [1.0] * 4


class FakeGraphRepository:
    def __init__(self):
        self.saved_graphs = []
        self.existing_by_type = {}

    async def save_graph(self, graph):
        self.saved_graphs.append(graph)

    async def list_entities_by_type(self, user_id: str, entity_type: str):
        return self.existing_by_type.get(entity_type, [])
