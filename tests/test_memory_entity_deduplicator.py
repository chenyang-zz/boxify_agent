import json

import pytest

from app.domain.models.memory_graph import EntityNode
from app.domain.services.memory.entity_deduplicator import MemoryEntityDeduplicator


@pytest.mark.anyio
async def test_deduplicator_merges_exact_batch_entities_without_llm():
    llm = FakeLLM()
    deduplicator = MemoryEntityDeduplicator(
        llm=llm,
        json_parser=FakeJSONParser(),
    )
    first = EntityNode(
        id="entity-user-1",
        user_id="user-a",
        name="用户",
        type="生命体",
        description="当前用户",
        importance=0.6,
        confidence=0.8,
        mention_count=1,
    )
    duplicate = EntityNode(
        id="entity-user-2",
        user_id="user-a",
        name="用户",
        type="生命体",
        description="当前正在使用系统的用户",
        importance=0.9,
        confidence=0.7,
        mention_count=2,
    )

    result = await deduplicator.dedup_batch({1: first, 2: duplicate})

    assert llm.calls == []
    assert len(result.entities) == 1
    assert result.entity_by_idx[1].id == result.entity_by_idx[2].id
    canonical = result.entities[0]
    assert canonical.description == "当前正在使用系统的用户"
    assert canonical.importance == 0.9
    assert canonical.confidence == 0.8
    assert canonical.mention_count == 3
    assert result.redirects == {"entity-user-2": "entity-user-1"}


@pytest.mark.anyio
async def test_deduplicator_uses_llm_for_similar_batch_entities():
    llm = FakeLLM(
        {
            "same_entity": True,
            "canonical_idx": 1,
            "confidence": 0.91,
            "reason": "同一个音乐人",
        }
    )
    deduplicator = MemoryEntityDeduplicator(
        llm=llm,
        json_parser=FakeJSONParser(),
    )
    first = EntityNode(
        id="entity-jay-cn",
        user_id="user-a",
        name="周杰伦",
        type="生命体",
        description="歌手",
        embedding=[1.0, 0.0],
    )
    second = EntityNode(
        id="entity-jay-tw",
        user_id="user-a",
        name="周杰倫",
        type="生命体",
        description="华语音乐人",
        embedding=[0.99, 0.01],
    )

    result = await deduplicator.dedup_batch({1: first, 2: second})

    assert len(llm.calls) == 1
    assert len(result.entities) == 1
    assert result.entity_by_idx[1].id == result.entity_by_idx[2].id
    assert result.redirects == {"entity-jay-tw": "entity-jay-cn"}


@pytest.mark.anyio
async def test_deduplicator_keeps_low_confidence_or_failed_llm_candidates_separate():
    low_confidence = MemoryEntityDeduplicator(
        llm=FakeLLM(
            {
                "same_entity": True,
                "canonical_idx": 1,
                "confidence": 0.42,
                "reason": "不确定",
            }
        ),
        json_parser=FakeJSONParser(),
    )
    first = EntityNode(
        id="entity-a",
        user_id="user-a",
        name="小明",
        type="生命体",
        embedding=[1.0, 0.0],
    )
    second = EntityNode(
        id="entity-b",
        user_id="user-a",
        name="小铭",
        type="生命体",
        embedding=[0.99, 0.01],
    )

    low_result = await low_confidence.dedup_batch({1: first, 2: second})

    assert len(low_result.entities) == 2
    failing = MemoryEntityDeduplicator(
        llm=FailingLLM(),
        json_parser=FakeJSONParser(),
    )

    failed_result = await failing.dedup_batch({1: first, 2: second})

    assert len(failed_result.entities) == 2


@pytest.mark.anyio
async def test_deduplicator_reuses_graph_entity_and_inherits_dynamic_fields():
    llm = FakeLLM(
        {
            "same_entity": True,
            "canonical_idx": 0,
            "confidence": 0.93,
            "reason": "别名指向同一个人",
        }
    )
    deduplicator = MemoryEntityDeduplicator(
        llm=llm,
        json_parser=FakeJSONParser(),
    )
    incoming = EntityNode(
        id="incoming-jay",
        user_id="user-a",
        name="杰伦",
        type="生命体",
        description="歌手",
        embedding=[0.99, 0.01],
        mention_count=1,
        memory_layer="short_term",
    )
    existing = EntityNode(
        id="existing-jay",
        user_id="user-a",
        name="周杰伦",
        type="生命体",
        description="长期关注的华语音乐人",
        embedding=[1.0, 0.0],
        mention_count=5,
        access_count=2,
        memory_layer="long_term",
        core_facts=["用户喜欢他的音乐"],
        traits=["华语流行"],
    )
    repository = FakeGraphRepository({"生命体": [existing]})

    result = await deduplicator.merge_with_graph("user-a", {7: incoming}, repository)

    canonical = result.entity_by_idx[7]
    assert canonical.id == "existing-jay"
    assert canonical.mention_count == 6
    assert canonical.access_count == 2
    assert canonical.memory_layer == "long_term"
    assert canonical.core_facts == ["用户喜欢他的音乐"]
    assert canonical.traits == ["华语流行"]
    assert canonical.description == "长期关注的华语音乐人"
    assert result.redirects == {"incoming-jay": "existing-jay"}


class FakeLLM:
    def __init__(self, payload=None):
        self._payload = payload or {}
        self.calls = []

    async def invoke(
        self,
        messages,
        tools=None,
        response_format=None,
        tool_choice=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "response_format": response_format,
                "tool_choice": tool_choice,
            }
        )
        return {"content": json.dumps(self._payload, ensure_ascii=False)}

    @property
    def model_name(self):
        return "fake-llm"

    @property
    def temperature(self):
        return 0

    @property
    def max_tokens(self):
        return 1024


class FailingLLM(FakeLLM):
    async def invoke(
        self,
        messages,
        tools=None,
        response_format=None,
        tool_choice=None,
    ):
        raise RuntimeError("llm unavailable")


class FakeJSONParser:
    async def invoke(self, text, default_value=None):
        return json.loads(text)


class FakeGraphRepository:
    def __init__(self, existing_by_type):
        self.existing_by_type = existing_by_type

    async def list_entities_by_type(self, user_id: str, entity_type: str):
        return self.existing_by_type.get(entity_type, [])
