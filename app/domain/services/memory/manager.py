from typing import Callable, Protocol

from app.domain.models.long_term_memory import LongTermMemory, MemorySource
from app.domain.repositories.vow import IUnitOfWork


class MemorySearch(Protocol):
    """长期记忆检索能力。"""

    async def search(self, query: str, top_k: int) -> list[LongTermMemory]:
        """检索长期记忆。"""
        ...


class LongTermMemoryManager:
    """当前用户长期记忆管理器。"""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork], user_id: str) -> None:
        self._uow_factory = uow_factory
        self._user_id = user_id

    async def remember_text(
        self,
        content: str,
        source: MemorySource,
        source_session_id: str | None = None,
    ) -> LongTermMemory:
        """保存一条当前用户长期记忆。"""
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空")

        memory = LongTermMemory(
            user_id=self._user_id,
            content=content,
            source=source,
            source_session_id=source_session_id,
        )
        memory.mark_completed(summary=content, keywords=self.extract_keywords(content))
        async with self._uow_factory() as uow:
            await uow.memory.save(memory)
        return memory

    async def list_memories(
        self, page: int, page_size: int
    ) -> tuple[list[LongTermMemory], int]:
        """分页读取当前用户记忆。"""
        async with self._uow_factory() as uow:
            return await uow.memory.list_by_user(self._user_id, page, page_size)

    async def search(self, query: str, top_k: int) -> list[LongTermMemory]:
        """检索当前用户长期记忆。"""
        query = query.strip()
        if not query:
            raise ValueError("检索关键词不能为空")
        top_k = max(1, min(top_k, 20))
        async with self._uow_factory() as uow:
            return await uow.memory.search_by_user(self._user_id, query, top_k)

    async def delete_memory(self, memory_id: str) -> bool:
        """删除当前用户记忆，返回是否删除成功。"""
        async with self._uow_factory() as uow:
            return await uow.memory.delete_by_user(self._user_id, memory_id)

    @classmethod
    def extract_keywords(cls, content: str) -> list[str]:
        """提取轻量关键词，给 V1 检索和响应展示使用。"""
        normalized = content.replace("，", " ").replace("。", " ").replace(",", " ")
        return [word for word in normalized.split() if word][:8]
