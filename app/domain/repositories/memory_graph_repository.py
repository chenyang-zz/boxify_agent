from typing import Protocol

from app.domain.models.memory_graph import (
    CommunityMemberResult,
    CommunityRelationResult,
    CommunityResult,
    CommunityVoteEntity,
    CommunityVoteNeighbor,
    EntityNode,
    InsightResult,
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

    async def reflection_top_entities(
        self, user_id: str, top_k: int
    ) -> list[EntityNode]:
        """返回反思使用的长期实体。"""
        ...

    async def reflection_entity_statements(
        self, user_id: str, entity_id: str, limit: int
    ) -> list[str]:
        """返回反思使用的代表性陈述。"""
        ...

    async def upsert_insight(
        self,
        user_id: str,
        theme: str,
        content: str,
        embedding: list[float] | None,
        importance: float,
        confidence: float,
        source_count: int,
        entity_ids: list[str],
    ) -> str:
        """按主题新增或更新高层洞察并重建实体溯源边。"""
        ...

    async def search_insights_by_vector(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[InsightResult]:
        """按向量检索当前用户洞察。"""
        ...

    async def list_insights(self, user_id: str) -> list[InsightResult]:
        """列出当前用户洞察。"""
        ...

    async def count_insights(self, user_id: str) -> int:
        """统计当前用户洞察数量。"""
        ...

    async def has_communities(self, user_id: str) -> bool:
        """判断当前用户是否已有社区。"""
        ...

    async def dialogue_entity_ids(self, user_id: str, dialogue_id: str) -> list[str]:
        """读取一次记忆萃取写入后关联的实体 ID。"""
        ...

    async def community_vote_entities(
        self, user_id: str, entity_ids: list[str] | None = None
    ) -> list[CommunityVoteEntity]:
        """读取社区聚类投票所需实体。"""
        ...

    async def community_vote_neighbors(
        self, user_id: str, entity_ids: list[str]
    ) -> dict[str, list[CommunityVoteNeighbor]]:
        """读取实体一跳邻居及其社区标签，用于 LPA 投票。"""
        ...

    async def upsert_community(self, user_id: str, community_id: str) -> None:
        """创建或保留社区节点。"""
        ...

    async def assign_entity_community(
        self, user_id: str, entity_id: str, community_id: str
    ) -> None:
        """将实体归入社区。"""
        ...

    async def refresh_community_member_count(
        self, user_id: str, community_id: str
    ) -> int:
        """刷新并返回社区成员数。"""
        ...

    async def community_members(
        self, user_id: str, community_id: str
    ) -> list[CommunityMemberResult]:
        """读取社区成员实体。"""
        ...

    async def community_relationships(
        self, user_id: str, community_id: str
    ) -> list[CommunityRelationResult]:
        """读取社区内部关系事实。"""
        ...

    async def update_community_metadata(
        self, user_id: str, community_id: str, name: str, summary: str
    ) -> None:
        """更新社区名称和摘要。"""
        ...

    async def list_communities(self, user_id: str) -> list[CommunityResult]:
        """列出当前用户社区。"""
        ...

    async def prune_empty_communities(self, user_id: str) -> None:
        """清理当前用户空社区。"""
        ...
