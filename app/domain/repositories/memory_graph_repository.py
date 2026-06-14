from typing import Protocol

from app.domain.models.memory_graph import EntityNode, MemoryGraph, MemoryGraphResult


class MemoryGraphRepository(Protocol):
    """长期记忆图谱仓储协议。"""

    async def save_graph(self, graph: MemoryGraph) -> None:
        """保存记忆图谱。"""
        ...

    async def list_entities_by_type(
        self, user_id: str, entity_type: str
    ) -> list[EntityNode]:
        """按用户和类型列出已有实体，用于写图前去重融合。"""
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
