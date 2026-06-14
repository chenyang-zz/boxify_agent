import pytest

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.application.services.memory_service import MemoryService
from app.domain.models.long_term_memory import LongTermMemory, MemorySource, MemoryStatus
from app.domain.models.memory_graph import (
    GraphRelationFact,
    MemoryGraphResult,
)
from app.domain.services.memory import LongTermMemoryManager


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


class MemoryUnitOfWork:
    def __init__(self, memory_repository):
        self.memory = memory_repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
