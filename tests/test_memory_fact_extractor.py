import pytest

from app.domain.models.memory_graph import ChunkNode, StatementNode
from app.domain.services.memory.fact_extractor import MemoryFactExtractor
from app.domain.services.prompts.memory import (
    DEDUP_ENTITY_PROMPT,
    EXTRACT_STATEMENTS_PROMPT,
    EXTRACT_TRIPLETS_PROMPT,
)
from app.infrastructure.external.json_parser.repair_json_parser import RepairJSONParser


def test_memory_graph_prompts_format_required_placeholders():
    statements_prompt = EXTRACT_STATEMENTS_PROMPT.format(
        text="用户喜欢周杰伦。",
        dialog_at="2026-06-16T09:00:00",
    )
    triplets_prompt = EXTRACT_TRIPLETS_PROMPT.format(
        statements=["用户喜欢周杰伦。"],
        dialog_at="2026-06-16T09:00:00",
    )

    assert "用户喜欢周杰伦。" in statements_prompt
    assert "用户喜欢周杰伦。" in triplets_prompt
    assert '{"statements"' in statements_prompt
    assert '"triplets"' in triplets_prompt
    assert "statement_type" in statements_prompt
    assert "has_unsolved_reference" in statements_prompt
    assert "valid_at" in statements_prompt
    assert "invalid_at" in statements_prompt
    assert "2026-06-16T09:00:00" in statements_prompt
    assert "entities" in triplets_prompt
    assert "subject_id" in triplets_prompt
    assert "valid_at" in triplets_prompt
    assert "invalid_at" in triplets_prompt
    assert '"events"' in triplets_prompt
    assert "一次性、过去发生" in triplets_prompt
    assert "稳定画像/偏好/关系不落 Event" in triplets_prompt
    assert "生命体" in triplets_prompt
    assert "偏好" in triplets_prompt

    dedup_prompt = DEDUP_ENTITY_PROMPT.format(
        left_idx=1,
        right_idx=2,
        left_name="周杰伦",
        left_type="生命体",
        left_description="歌手",
        right_name="周杰倫",
        right_type="生命体",
        right_description="华语音乐人",
        name_similarity="0.900",
        embedding_similarity="0.950",
        name_contains="false",
    )
    assert "same_entity" in dedup_prompt
    assert "canonical_idx" in dedup_prompt
    assert "拿不准" in dedup_prompt
    assert "低于 0.8 不会合并" in dedup_prompt


@pytest.mark.anyio
async def test_memory_fact_extractor_extracts_and_filters_statements():
    extractor = MemoryFactExtractor(
        llm=FakeLLM(
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
                            "valid_at": None,
                            "invalid_at": None,
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
                            "statement": "",
                            "has_unsolved_reference": False,
                        },
                    ]
                }
            ]
        ),
        json_parser=RepairJSONParser(),
    )

    statements = await extractor.extract_statements(
        [
            ChunkNode(
                id="chunk-1",
                user_id="user-a",
                dialogue_id="dialogue-1",
                index=0,
                text="用户住在上海。",
            )
        ]
    )

    assert [statement.text for statement in statements] == ["用户住在上海。"]
    assert statements[0].user_id == "user-a"
    assert statements[0].chunk_id == "chunk-1"
    assert statements[0].statement_type == "FACT"
    assert statements[0].importance == 0.7
    assert statements[0].valid_at is None
    assert statements[0].invalid_at is None


@pytest.mark.anyio
async def test_memory_fact_extractor_infers_dynamic_statement_valid_at():
    extractor = MemoryFactExtractor(
        llm=FakeLLM(
            [
                {
                    "statements": [
                        {
                            "statement": "用户现在在腾讯工作。",
                            "statement_type": "FACT",
                            "temporal_type": "DYNAMIC",
                            "has_unsolved_reference": False,
                            "importance": 0.8,
                            "confidence": 0.9,
                            "valid_at": "NULL",
                            "invalid_at": "",
                        }
                    ]
                }
            ]
        ),
        json_parser=RepairJSONParser(),
    )

    statements = await extractor.extract_statements(
        [
            ChunkNode(
                id="chunk-1",
                user_id="user-a",
                dialogue_id="dialogue-1",
                index=0,
                text="用户现在在腾讯工作。",
            )
        ],
        dialog_at="2026-06-16T09:00:00",
    )

    assert statements[0].valid_at.isoformat() == "2026-06-16T09:00:00"
    assert statements[0].invalid_at is None


@pytest.mark.anyio
async def test_memory_fact_extractor_extracts_triplets():
    extractor = MemoryFactExtractor(
        llm=FakeLLM(
            [
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
                            "valid_at": "2026-06-16T09:00:00",
                            "invalid_at": None,
                        }
                    ],
                    "events": [
                        {
                            "title": "参加周杰伦演唱会",
                            "description": "用户昨天参加了周杰伦演唱会",
                            "event_time": "2026-06-15T20:00:00",
                            "participants": ["用户", "周杰伦"],
                        }
                    ],
                }
            ]
        ),
        json_parser=RepairJSONParser(),
    )

    triplets = await extractor.extract_triplets(
        [
            StatementNode(
                id="statement-1",
                user_id="user-a",
                chunk_id="chunk-1",
                index=0,
                text="用户喜欢周杰伦。",
            )
        ]
    )

    assert [entity.name for entity in triplets.entities] == ["用户", "周杰伦"]
    assert triplets.triplets[0].subject_id == 1
    assert triplets.triplets[0].predicate == "偏好"
    assert triplets.triplets[0].object_id == 2
    assert triplets.triplets[0].valid_at == "2026-06-16T09:00:00"
    assert triplets.triplets[0].invalid_at is None
    assert triplets.events[0].title == "参加周杰伦演唱会"
    assert triplets.events[0].participants == ["用户", "周杰伦"]


@pytest.mark.anyio
async def test_memory_fact_extractor_uses_safe_empty_defaults_for_bad_json():
    extractor = MemoryFactExtractor(
        llm=FakeLLM(["not-json", "not-json"]),
        json_parser=RepairJSONParser(),
    )

    statements = await extractor.extract_statements(
        [
            ChunkNode(
                id="chunk-1",
                user_id="user-a",
                dialogue_id="dialogue-1",
                index=0,
                text="用户住在上海。",
            )
        ]
    )
    triplets = await extractor.extract_triplets(
        [
            StatementNode(
                id="statement-1",
                user_id="user-a",
                chunk_id="chunk-1",
                index=0,
                text="用户喜欢周杰伦。",
            )
        ]
    )

    assert statements == []
    assert triplets.entities == []
    assert triplets.triplets == []
    assert triplets.events == []


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
