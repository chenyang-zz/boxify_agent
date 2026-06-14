from typing import Protocol

from app.domain.models.memory_graph import MemoryGraph, MemoryGraphResult


class MemoryGraphRepository(Protocol):
    """长期记忆图谱仓储协议。"""

    async def save_graph(self, graph: MemoryGraph) -> None:
        """保存记忆图谱。"""
        ...

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryGraphResult]:
        """检索用户记忆图谱。"""
        ...
