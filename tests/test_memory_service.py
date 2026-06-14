import pytest

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.application.services.memory_service import MemoryService
from app.domain.models.long_term_memory import LongTermMemory, MemorySource
from app.domain.services.memory import LongTermMemoryManager


@pytest.mark.anyio
async def test_long_term_memory_manager_remembers_text_for_current_user():
    repository = InMemoryMemoryRepository()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    memory = await manager.remember_text("我喜欢周杰伦的歌", source=MemorySource.MANUAL)

    assert memory.user_id == "user-a"
    assert memory.source == MemorySource.MANUAL
    assert memory.content == "我喜欢周杰伦的歌"
    assert memory.status == "completed"
    assert repository.saved[0] == memory


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
async def test_long_term_memory_manager_remembers_session_source():
    repository = InMemoryMemoryRepository()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    memory = await manager.remember_text(
        "用户喜欢安静的工作环境",
        source=MemorySource.SESSION,
        source_session_id="session-1",
    )

    assert memory.source == MemorySource.SESSION
    assert memory.source_session_id == "session-1"
    assert memory.keywords == ["用户喜欢安静的工作环境"]


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


class MemoryUnitOfWork:
    def __init__(self, memory_repository):
        self.memory = memory_repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
