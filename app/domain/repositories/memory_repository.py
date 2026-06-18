from abc import ABC, abstractmethod

from app.domain.models.long_term_memory import LongTermMemory
from app.domain.models.memory_graph import MemoryQualityFailedMemoryResult


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
    async def get_user_id_by_memory_id(self, memory_id: str) -> str | None:
        """按记忆 ID 读取所属用户 ID，供异步任务定位配置。"""
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

    @abstractmethod
    async def status_counts(self, user_id: str) -> dict[str, int]:
        """统计当前用户各处理状态的长期记忆数量。"""
        ...

    @abstractmethod
    async def recent_failed(
        self, user_id: str, limit: int
    ) -> list[MemoryQualityFailedMemoryResult]:
        """读取当前用户最近失败的记忆摘要。"""
        ...
