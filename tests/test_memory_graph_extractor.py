import pytest

from app.domain.services.memory.graph_extractor import MemoryGraphExtractor
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
    assert '{"triplets"' in triplets_prompt


@pytest.mark.anyio
async def test_memory_graph_extractor_writes_four_layer_graph_with_deduped_entities():
    llm = FakeLLM(
        [
            {
                "statements": [
                    {"text": "用户住在上海。"},
                    {"text": "用户喜欢周杰伦。"},
                ]
            },
            {
                "triplets": [
                    {
                        "head": {
                            "name": "用户",
                            "type": "Person",
                            "description": "当前用户",
                        },
                        "relation": "LIVES_IN",
                        "tail": {
                            "name": "上海",
                            "type": "Location",
                            "description": "城市",
                        },
                        "evidence": "用户住在上海。",
                    },
                    {
                        "head": {
                            "name": "用户",
                            "type": "Person",
                            "description": "当前用户",
                        },
                        "relation": "LIKES",
                        "tail": {
                            "name": "周杰伦",
                            "type": "Person",
                            "description": "歌手",
                        },
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
    assert {(entity.name, entity.type) for entity in graph.entities} == {
        ("用户", "Person"),
        ("上海", "Location"),
        ("周杰伦", "Person"),
    }
    assert all(entity.embedding for entity in graph.entities)
    assert [(rel.name, rel.evidence) for rel in graph.relations] == [
        ("LIVES_IN", "用户住在上海。"),
        ("LIKES", "用户喜欢周杰伦。"),
    ]
    assert stats == {
        "dialogue_id": graph.dialogue.id,
        "chunks": 1,
        "statements": 2,
        "entities": 3,
        "relations": 2,
    }


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
    assert stats["dialogue_id"] == graph.dialogue.id


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

    async def save_graph(self, graph):
        self.saved_graphs.append(graph)
