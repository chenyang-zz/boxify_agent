from typing import Protocol

from app.domain.models.memory_graph import (
    EntityNode,
    MemoryGraph,
    MemoryGraphResult,
    MemoryPromotionStats,
)


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

    async def bump_entity_access(self, user_id: str, entity_ids: list[str]) -> None:
        """记录实体被检索命中的访问动态。"""
        ...

    async def promote_short_to_long(
        self,
        user_id: str,
        min_access: int,
        min_importance: float,
        min_mention: int,
        age_before: str,
    ) -> MemoryPromotionStats:
        """按动力学阈值把短期图谱记忆提升为长期。"""
        ...

    async def top_long_term_entities(
        self, user_id: str, top_k: int
    ) -> list[EntityNode]:
        """返回最适合画像增强的长期实体。"""
        ...

    async def entity_statements(self, user_id: str, entity_id: str) -> list[str]:
        """返回实体关联的原子陈述文本。"""
        ...

    async def write_entity_profile(
        self,
        user_id: str,
        entity_id: str,
        core_facts: list[str],
        traits: list[str],
    ) -> None:
        """回写长期实体画像摘要。"""
        ...
