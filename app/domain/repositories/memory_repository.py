from abc import ABC, abstractmethod

from app.domain.models.long_term_memory import LongTermMemory


class MemoryRepository(ABC):
    """长期记忆仓储接口。"""

    @abstractmethod
    async def save(self, memory: LongTermMemory) -> None:
        """保存记忆条目。"""
        ...

    @abstractmethod
    async def get_by_user(self, user_id: str, memory_id: str) -> LongTermMemory | None:
        """按用户边界读取记忆。"""
        ...

    @abstractmethod
    async def list_by_user(
        self, user_id: str, page: int, page_size: int
    ) -> tuple[list[LongTermMemory], int]:
        """分页列出用户记忆。"""
        ...

    @abstractmethod
    async def search_by_user(
        self, user_id: str, query: str, top_k: int
    ) -> list[LongTermMemory]:
        """在用户记忆中检索。"""
        ...

    @abstractmethod
    async def delete_by_user(self, user_id: str, memory_id: str) -> bool:
        """删除用户记忆，返回是否删除成功。"""
        ...
