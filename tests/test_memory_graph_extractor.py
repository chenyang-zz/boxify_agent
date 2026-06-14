import pytest

from app.domain.services.memory.graph_extractor import MemoryGraphExtractor
from app.domain.models.memory_graph import EntityNode, MemoryGraphStats
from app.domain.services.prompts.memory import (
    EXTRACT_STATEMENTS_PROMPT,
    EXTRACT_TRIPLETS_PROMPT,
)
from app.infrastructure.external.json_parser.repair_json_parser import RepairJSONParser


def test_memory_graph_prompts_format_required_placeholders():
    statements_prompt = EXTRACT_STATEMENTS_PROMPT.format(text="用户喜欢周杰伦。")
    triplets_prompt = EXTRACT_TRIPLETS_PROMPT.format(statements=["用户喜欢周杰伦。"])

    assert "用户喜欢周杰伦。" in statements_prompt
    assert "用户喜欢周杰伦。" in triplets_prompt
    assert '{"statements"' in statements_prompt
    assert '"triplets"' in triplets_prompt
    assert "statement_type" in statements_prompt
    assert "has_unsolved_reference" in statements_prompt
    assert "entities" in triplets_prompt
    assert "subject_id" in triplets_prompt
    assert "生命体" in triplets_prompt
    assert "偏好" in triplets_prompt


@pytest.mark.anyio
async def test_memory_graph_extractor_writes_four_layer_graph_with_deduped_entities():
    llm = FakeLLM(
        [
            {
                "statements": [
                    {
                        "statement": "用户住在上海。",
                        "statement_type": "FACT",
                        "temporal_type": "STATIC",
                        "has_unsolved_reference": False,
                        "importance": 0.7,
                        "confidence": 0.9,
                    },
                    {
                        "statement": "他喜欢模糊对象。",
                        "statement_type": "FACT",
                        "temporal_type": "STATIC",
                        "has_unsolved_reference": True,
                        "importance": 0.5,
                        "confidence": 0.4,
                    },
                    {
                        "statement": "用户喜欢周杰伦。",
                        "statement_type": "FACT",
                        "temporal_type": "STATIC",
                        "has_unsolved_reference": False,
                        "importance": 0.8,
                        "confidence": 0.95,
                    },
                ]
            },
            {
                "entities": [
                    {
                        "entity_idx": 1,
                        "name": "用户",
                        "type": "生命体",
                        "description": "当前用户",
                    },
                    {
                        "entity_idx": 2,
                        "name": "上海",
                        "type": "地点设施",
                        "description": "城市",
                    },
                    {
                        "entity_idx": 3,
                        "name": "用户",
                        "type": "生命体",
                        "description": "当前用户",
                    },
                    {
                        "entity_idx": 4,
                        "name": "周杰伦",
                        "type": "Person",
                        "description": "歌手",
                    },
                ],
                "triplets": [
                    {
                        "subject_id": 1,
                        "predicate": "位于",
                        "object_id": 2,
                        "evidence": "用户住在上海。",
                    },
                    {
                        "subject_id": 3,
                        "predicate": "LIKES",
                        "object_id": 4,
                        "evidence": "用户喜欢周杰伦。",
                    },
                ]
            },
        ]
    )
    embedding = FakeEmbedding()
    repository = FakeGraphRepository()
    extractor = MemoryGraphExtractor(
        llm=llm,
        embedding=embedding,
        json_parser=RepairJSONParser(),
        graph_repository=repository,
    )

    stats = await extractor.extract_memory(
        memory_id="mem-1",
        user_id="user-a",
        content="用户住在上海。用户喜欢周杰伦。",
    )

    graph = repository.saved_graphs[0]
    assert graph.dialogue.memory_id == "mem-1"
    assert graph.dialogue.user_id == "user-a"
    assert len(graph.chunks) == 1
    assert [statement.text for statement in graph.statements] == [
        "用户住在上海。",
        "用户喜欢周杰伦。",
    ]
    assert [statement.statement_type for statement in graph.statements] == [
        "FACT",
        "FACT",
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
    assert graph.relations[0].source_entity_id == graph.relations[1].source_entity_id
    assert stats == MemoryGraphStats(
        dialogue_id=graph.dialogue.id,
        chunks=1,
        statements=2,
        entities=3,
        relations=2,
    )


@pytest.mark.anyio
async def test_memory_graph_extractor_uses_safe_empty_defaults_for_bad_json():
    llm = FakeLLM(["not-json", "not-json"])
    repository = FakeGraphRepository()
    extractor = MemoryGraphExtractor(
        llm=llm,
        embedding=FakeEmbedding(),
        json_parser=RepairJSONParser(),
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
    llm = FakeLLM(
        [
            {
                "statements": [
                    {
                        "statement": "用户喜欢周杰伦。",
                        "statement_type": "FACT",
                        "temporal_type": "STATIC",
                        "has_unsolved_reference": False,
                    }
                ]
            },
            {
                "entities": [
                    {
                        "entity_idx": 1,
                        "name": "用户",
                        "type": "生命体",
                        "description": "当前用户",
                    },
                    {
                        "entity_idx": 2,
                        "name": "周杰伦",
                        "type": "生命体",
                        "description": "歌手",
                    },
                ],
                "triplets": [
                    {
                        "subject_id": 1,
                        "predicate": "偏好",
                        "object_id": 2,
                        "evidence": "用户喜欢周杰伦。",
                    }
                ],
            },
        ]
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
        llm=llm,
        embedding=FakeEmbedding(),
        json_parser=RepairJSONParser(),
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


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls.append(messages)
        content = self.responses.pop(0)
        return {"content": content}

    @property
    def model_name(self):
        return "fake-llm"

    @property
    def temperature(self):
        return 0

    @property
    def max_tokens(self):
        return 1024


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
