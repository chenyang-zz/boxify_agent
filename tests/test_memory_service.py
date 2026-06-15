import pytest

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.application.services.memory_service import MemoryService
from app.domain.models.long_term_memory import LongTermMemory, MemorySource, MemoryStatus
from app.domain.models.memory_graph import (
    EntityNode,
    GraphRelationFact,
    MemoryConsolidationStats,
    MemoryGraphResult,
    MemoryPromotionStats,
)
from app.domain.services.memory import LongTermMemoryManager, MemoryConsolidator
from app.domain.services.memory.profile_summarizer import MemoryProfileSummarizer


@pytest.mark.anyio
async def test_long_term_memory_manager_remembers_text_for_current_user():
    repository = InMemoryMemoryRepository()
    dispatcher = FakeTaskDispatcher()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        task_dispatcher=dispatcher,
    )

    memory = await manager.remember_text("我喜欢周杰伦的歌", source=MemorySource.MANUAL)

    assert memory.user_id == "user-a"
    assert memory.source == MemorySource.MANUAL
    assert memory.content == "我喜欢周杰伦的歌"
    assert memory.status == MemoryStatus.PENDING
    assert memory.summary == "我喜欢周杰伦的歌"
    assert repository.saved[0] == memory
    assert dispatcher.extract_memory_calls == [memory.id]


@pytest.mark.anyio
async def test_long_term_memory_manager_search_is_scoped_to_current_user():
    repository = InMemoryMemoryRepository()
    user_a_memory = LongTermMemory(
        user_id="user-a",
        content="我喜欢周杰伦的歌",
        summary="用户喜欢周杰伦的歌",
    )
    user_b_memory = LongTermMemory(
        user_id="user-b",
        content="我喜欢林俊杰的歌",
        summary="用户喜欢林俊杰的歌",
    )
    await repository.save(user_a_memory)
    await repository.save(user_b_memory)
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    results = await manager.search("周杰伦", top_k=10)

    assert results == [user_a_memory]


@pytest.mark.anyio
async def test_long_term_memory_manager_prefers_graph_results():
    repository = InMemoryMemoryRepository()
    graph_repository = FakeGraphRepository(
        [
            MemoryGraphResult(
                entity_id="entity-1",
                entity_name="周杰伦",
                entity_type="Person",
                description="歌手",
                score=0.91,
                source_memory_id="mem-1",
                source_memory_summary="用户喜欢周杰伦的歌",
                relations=[
                    GraphRelationFact(
                        name="LIKES",
                        direction="incoming",
                        neighbor_name="用户",
                        neighbor_type="Person",
                        evidence="用户喜欢周杰伦的歌",
                    )
                ],
            )
        ]
    )
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=graph_repository,
        embedding=FakeEmbedding(),
    )

    results = await manager.search("喜欢的歌手", top_k=3)

    assert len(results) == 1
    assert results[0].content == "用户喜欢周杰伦的歌"
    assert results[0].graph_data is not None
    assert results[0].graph_data.entity_name == "周杰伦"
    assert results[0].graph_data.relations[0].evidence == "用户喜欢周杰伦的歌"
    assert graph_repository.calls == [("user-a", "喜欢的歌手", 3, [1.0])]


@pytest.mark.anyio
async def test_long_term_memory_manager_falls_back_to_pg_when_graph_has_no_result():
    repository = InMemoryMemoryRepository()
    pg_memory = LongTermMemory(
        user_id="user-a",
        content="我喜欢周杰伦的歌",
        summary="用户喜欢周杰伦",
    )
    await repository.save(pg_memory)
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=FakeGraphRepository([]),
        embedding=FakeEmbedding(),
    )

    results = await manager.search("周杰伦", top_k=3)

    assert results == [pg_memory]


@pytest.mark.anyio
async def test_long_term_memory_manager_falls_back_to_pg_when_graph_errors():
    repository = InMemoryMemoryRepository()
    pg_memory = LongTermMemory(
        user_id="user-a",
        content="我喜欢周杰伦的歌",
        summary="用户喜欢周杰伦",
    )
    await repository.save(pg_memory)
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=ExplodingGraphRepository(),
        embedding=FakeEmbedding(),
    )

    results = await manager.search("周杰伦", top_k=3)

    assert results == [pg_memory]


@pytest.mark.anyio
async def test_long_term_memory_manager_uses_graph_fulltext_when_embedding_fails():
    repository = InMemoryMemoryRepository()
    graph_repository = FakeGraphRepository(
        [
            MemoryGraphResult(
                entity_id="entity-1",
                entity_name="周杰伦",
                entity_type="生命体",
                description="歌手",
                score=0.7,
                source_memory_summary="用户喜欢周杰伦",
            )
        ]
    )
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=graph_repository,
        embedding=ExplodingEmbedding(),
    )

    results = await manager.search("周杰伦", top_k=3)

    assert results[0].graph_data is not None
    assert results[0].graph_data.entity_name == "周杰伦"
    assert graph_repository.calls == [("user-a", "周杰伦", 3, None)]


@pytest.mark.anyio
async def test_memory_consolidation_service_promotes_and_enhances_profiles():
    repository = FakeConsolidationGraphRepository()
    service = MemoryConsolidator(
        user_id="user-a",
        graph_repository=repository,
        profile_summarizer=FakeProfileSummarizer(
            core_facts=["用户长期喜欢周杰伦"],
            traits=["偏好华语流行"],
        ),
    )

    stats = await service.consolidate()

    assert stats == MemoryConsolidationStats(
        promoted_entities=2,
        promoted_statements=3,
        enhanced_profiles=1,
    )
    assert repository.promote_calls == [
        {
            "user_id": "user-a",
            "min_access": 3,
            "min_importance": 0.8,
            "min_mention": 3,
        }
    ]
    assert repository.profile_writes == [
        (
            "user-a",
            "entity-1",
            ["用户长期喜欢周杰伦"],
            ["偏好华语流行"],
        )
    ]


@pytest.mark.anyio
async def test_memory_consolidation_service_skips_single_profile_failure():
    repository = FakeConsolidationGraphRepository(
        top_entities=[
            EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
            EntityNode(id="entity-2", user_id="user-a", name="林俊杰", type="生命体"),
        ]
    )
    service = MemoryConsolidator(
        user_id="user-a",
        graph_repository=repository,
        profile_summarizer=FlakyProfileSummarizer(),
    )

    stats = await service.consolidate()

    assert stats.enhanced_profiles == 1
    assert repository.profile_writes == [
        ("user-a", "entity-2", ["可用事实"], ["可用特质"])
    ]


@pytest.mark.anyio
async def test_memory_profile_summarizer_parses_profile_json():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM(
            '{"core_facts":["用户长期喜欢周杰伦"],"traits":["偏好华语流行"]}'
        ),
        json_parser=FakeJSONParser(),
    )

    core_facts, traits = await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    )

    assert core_facts == ["用户长期喜欢周杰伦"]
    assert traits == ["偏好华语流行"]


@pytest.mark.anyio
async def test_memory_profile_summarizer_falls_back_for_bad_json():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM("not-json"),
        json_parser=FakeJSONParser(),
    )

    assert await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    ) == ([], [])


@pytest.mark.anyio
async def test_memory_profile_summarizer_falls_back_for_non_dict_json():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM('["not-object"]'),
        json_parser=FakeJSONParser(),
    )

    assert await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    ) == ([], [])


@pytest.mark.anyio
async def test_memory_profile_summarizer_coerces_short_string_lists():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM(
            {
                "core_facts": ["  稳定事实  ", "", "x" * 250],
                "traits": ["偏好华语流行", None],
            }
        ),
        json_parser=FakeJSONParser(),
    )

    core_facts, traits = await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    )

    assert core_facts == ["稳定事实", "x" * 200]
    assert traits == ["偏好华语流行", "None"]


@pytest.mark.anyio
async def test_long_term_memory_manager_remembers_session_source():
    repository = InMemoryMemoryRepository()
    dispatcher = FakeTaskDispatcher()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        task_dispatcher=dispatcher,
    )

    memory = await manager.remember_text(
        "用户喜欢安静的工作环境",
        source=MemorySource.SESSION,
        source_session_id="session-1",
    )

    assert memory.source == MemorySource.SESSION
    assert memory.source_session_id == "session-1"
    assert memory.keywords == ["用户喜欢安静的工作环境"]
    assert dispatcher.extract_memory_calls == [memory.id]


@pytest.mark.anyio
async def test_long_term_memory_manager_delete_returns_false_when_missing():
    repository = InMemoryMemoryRepository()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    assert await manager.delete_memory("missing") is False


@pytest.mark.anyio
async def test_application_memory_service_converts_empty_content_to_bad_request():
    repository = InMemoryMemoryRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    with pytest.raises(BadRequestError) as exc:
        await service.remember_text("   ")

    assert exc.value.msg == "记忆内容不能为空"


@pytest.mark.anyio
async def test_application_memory_service_converts_empty_query_to_bad_request():
    repository = InMemoryMemoryRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    with pytest.raises(BadRequestError) as exc:
        await service.search("   ", top_k=5)

    assert exc.value.msg == "检索关键词不能为空"


@pytest.mark.anyio
async def test_application_memory_service_converts_missing_delete_to_not_found():
    repository = InMemoryMemoryRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    with pytest.raises(NotFoundError) as exc:
        await service.delete_memory("missing")

    assert exc.value.msg == "记忆不存在或无权访问"


class InMemoryMemoryRepository:
    def __init__(self):
        self.saved = []

    async def save(self, memory):
        self.saved.append(memory)

    async def get_by_user(self, user_id: str, memory_id: str):
        for memory in self.saved:
            if memory.user_id == user_id and memory.id == memory_id:
                return memory
        return None

    async def get_user_id_by_memory_id(self, memory_id: str):
        for memory in self.saved:
            if memory.id == memory_id:
                return memory.user_id
        return None

    async def list_by_user(self, user_id: str, page: int, page_size: int):
        memories = [memory for memory in self.saved if memory.user_id == user_id]
        return memories[(page - 1) * page_size : page * page_size], len(memories)

    async def search_by_user(self, user_id: str, query: str, top_k: int):
        return [
            memory
            for memory in self.saved
            if memory.user_id == user_id
            and (query in memory.content or (memory.summary and query in memory.summary))
        ][:top_k]

    async def delete_by_user(self, user_id: str, memory_id: str):
        before = len(self.saved)
        self.saved = [
            memory
            for memory in self.saved
            if not (memory.user_id == user_id and memory.id == memory_id)
        ]
        return len(self.saved) != before


class FakeTaskDispatcher:
    def __init__(self):
        self.extract_memory_calls = []

    async def dispatch_parse_document(self, document_id: str) -> None:
        raise AssertionError("memory tests should not dispatch document parsing")

    async def dispatch_extract_memory(self, memory_id: str) -> None:
        self.extract_memory_calls.append(memory_id)


class FakeGraphRepository:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def save_graph(self, graph):
        raise AssertionError("memory manager tests should not save graphs")

    async def search(self, user_id: str, query: str, top_k: int, query_embedding=None):
        self.calls.append((user_id, query, top_k, query_embedding))
        return self.results


class ExplodingGraphRepository:
    async def save_graph(self, graph):
        raise AssertionError("memory manager tests should not save graphs")

    async def search(self, user_id: str, query: str, top_k: int, query_embedding=None):
        raise RuntimeError("neo4j unavailable")


class FakeEmbedding:
    async def embed(self, texts):
        return [[1.0] for _ in texts]

    async def embed_one(self, text):
        return [1.0]

    @property
    def model_name(self):
        return "fake-embedding"


class ExplodingEmbedding:
    async def embed(self, texts):
        raise RuntimeError("embedding unavailable")

    async def embed_one(self, text):
        raise RuntimeError("embedding unavailable")

    @property
    def model_name(self):
        return "exploding-embedding"


class FakeConsolidationGraphRepository(FakeGraphRepository):
    def __init__(self, top_entities=None):
        super().__init__([])
        self.promote_calls = []
        self.profile_writes = []
        self.top_entities = top_entities or [
            EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体")
        ]

    async def promote_short_to_long(
        self, user_id, min_access, min_importance, min_mention, age_before
    ):
        self.promote_calls.append(
            {
                "user_id": user_id,
                "min_access": min_access,
                "min_importance": min_importance,
                "min_mention": min_mention,
            }
        )
        assert age_before
        return MemoryPromotionStats(promoted_entities=2, promoted_statements=3)

    async def top_long_term_entities(self, user_id, top_k):
        assert user_id == "user-a"
        assert top_k == 20
        return self.top_entities

    async def entity_statements(self, user_id, entity_id):
        assert user_id == "user-a"
        return ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"]

    async def write_entity_profile(self, user_id, entity_id, core_facts, traits):
        self.profile_writes.append((user_id, entity_id, core_facts, traits))


class FakeProfileSummarizer:
    def __init__(self, core_facts=None, traits=None):
        self.core_facts = core_facts or []
        self.traits = traits or []
        self.calls = []

    async def summarize(self, entity, statements):
        self.calls.append((entity.id, statements))
        return self.core_facts, self.traits


class FlakyProfileSummarizer:
    def __init__(self):
        self.calls = 0

    async def summarize(self, entity, statements):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("profile failed")
        return ["可用事实"], ["可用特质"]


class FakeProfileLLM:
    def __init__(self, content):
        self.content = content

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        assert response_format == {"type": "json_object"}
        assert "core_facts" in messages[-1]["content"]
        return {"content": self.content}

    @property
    def model_name(self):
        return "fake-profile-llm"

    @property
    def temperature(self):
        return 0

    @property
    def max_tokens(self):
        return 0


class FakeJSONParser:
    async def invoke(self, text, default_value=None):
        import json

        return json.loads(text)


class MemoryUnitOfWork:
    def __init__(self, memory_repository):
        self.memory = memory_repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
