import logging

from app.domain.models.memory_graph import (
    CommunityMember,
    CommunityRelationFact,
    CommunityVoteEntity,
    CommunityVoteNeighbor,
    MemoryCommunityClusterStats,
    stable_memory_graph_id,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.services.memory.community_summarizer import MemoryCommunitySummarizer
from app.utils.vector import average_vector, cosine_similarity
from core.config import get_settings

logger = logging.getLogger(__name__)


class MemoryCommunityClusterer:
    """使用 LPA 将记忆实体聚成主题社区。"""

    def __init__(
        self,
        user_id: str,
        graph_repository: MemoryGraphRepository,
        summarizer: MemoryCommunitySummarizer | None = None,
    ) -> None:
        self._user_id = user_id
        self._graph_repository = graph_repository
        self._summarizer = summarizer
        self._settings = get_settings()

    async def cluster(
        self, dialogue_id: str | None = None
    ) -> MemoryCommunityClusterStats:
        """执行全量或增量聚类。"""
        if dialogue_id and await self._graph_repository.has_communities(self._user_id):
            entity_ids = await self._graph_repository.dialogue_entity_ids(
                self._user_id, dialogue_id
            )
            if not entity_ids:
                return MemoryCommunityClusterStats(skipped="no_entities")
            return await self._incremental_cluster(entity_ids)
        return await self._full_cluster()

    async def _full_cluster(self) -> MemoryCommunityClusterStats:
        """全量 LPA 聚类。"""
        entities = await self._graph_repository.community_vote_entities(self._user_id)
        if not entities:
            return MemoryCommunityClusterStats(skipped="no_entities")
        entity_by_id = {entity.id: entity for entity in entities}
        labels = {entity.id: entity.community_id or entity.id for entity in entities}
        neighbors = await self._graph_repository.community_vote_neighbors(
            self._user_id, list(entity_by_id)
        )
        for _ in range(self._settings.memory_community_max_iterations):
            changed = 0
            for entity_id, entity in entity_by_id.items():
                voted = self._weighted_vote(
                    neighbors.get(entity_id, []),
                    getattr(entity, "embedding", []),
                    labels,
                )
                if voted and voted != labels[entity_id]:
                    labels[entity_id] = voted
                    changed += 1
            if changed == 0:
                break

        community_ids = await self._flush_labels(labels)
        merged = await self._merge_communities(community_ids)
        enhanced = await self._update_metadata(community_ids)
        await self._graph_repository.prune_empty_communities(self._user_id)
        return MemoryCommunityClusterStats(
            communities=len(community_ids) - merged,
            assigned_entities=len(labels),
            merged_communities=merged,
            enhanced_communities=enhanced,
        )

    async def _incremental_cluster(
        self, entity_ids: list[str]
    ) -> MemoryCommunityClusterStats:
        """只为新实体按邻居标签归入社区。"""
        entities = await self._graph_repository.community_vote_entities(
            self._user_id, entity_ids
        )
        if not entities:
            return MemoryCommunityClusterStats(skipped="no_entities")
        neighbors = await self._graph_repository.community_vote_neighbors(
            self._user_id, [entity.id for entity in entities]
        )
        touched: set[str] = set()
        assigned = 0
        for entity in entities:
            target = self._weighted_vote(
                neighbors.get(entity.id, []),
                getattr(entity, "embedding", []),
                labels={},
            )
            if not target:
                target = self._community_id_for_label(entity.id)
            await self._graph_repository.upsert_community(self._user_id, target)
            await self._graph_repository.assign_entity_community(
                self._user_id, entity.id, target
            )
            await self._graph_repository.refresh_community_member_count(
                self._user_id, target
            )
            touched.add(target)
            assigned += 1

        enhanced = await self._update_metadata(list(touched))
        return MemoryCommunityClusterStats(
            communities=len(touched),
            assigned_entities=assigned,
            enhanced_communities=enhanced,
        )

    async def _flush_labels(self, labels: dict[str, str]) -> list[str]:
        """把标签写入 Community 节点和 Entity 归属。"""
        community_by_label = {
            label: self._community_id_for_label(label) for label in set(labels.values())
        }
        for community_id in community_by_label.values():
            await self._graph_repository.upsert_community(self._user_id, community_id)
        for entity_id, label in labels.items():
            await self._graph_repository.assign_entity_community(
                self._user_id,
                entity_id,
                community_by_label[label],
            )
        for community_id in community_by_label.values():
            await self._graph_repository.refresh_community_member_count(
                self._user_id, community_id
            )
        return list(community_by_label.values())

    async def _merge_communities(self, community_ids: list[str]) -> int:
        """合并平均向量相似度高的社区。"""
        if len(community_ids) < 2:
            return 0
        members = {
            community_id: await self._graph_repository.community_members(
                self._user_id, community_id
            )
            for community_id in community_ids
        }
        avg_embeddings = {
            community_id: average_vector(
                [getattr(member, "embedding", None) for member in rows]
            )
            for community_id, rows in members.items()
        }
        sizes = {community_id: len(rows) for community_id, rows in members.items()}
        merged_into: dict[str, str] = {}

        def root(community_id: str) -> str:
            while community_id in merged_into:
                community_id = merged_into[community_id]
            return community_id

        for left_index, left in enumerate(community_ids):
            for right in community_ids[left_index + 1 :]:
                left_root = root(left)
                right_root = root(right)
                if left_root == right_root:
                    continue
                similarity = cosine_similarity(
                    avg_embeddings.get(left_root),
                    avg_embeddings.get(right_root),
                )
                if similarity <= self._settings.memory_community_merge_threshold:
                    continue
                keep, dissolve = (
                    (left_root, right_root)
                    if sizes.get(left_root, 0) >= sizes.get(right_root, 0)
                    else (right_root, left_root)
                )
                merged_into[dissolve] = keep
                for member in members.get(dissolve, []):
                    await self._graph_repository.assign_entity_community(
                        self._user_id,
                        _member_id(member),
                        keep,
                    )
                sizes[keep] = sizes.get(keep, 0) + sizes.get(dissolve, 0)
                sizes[dissolve] = 0
                await self._graph_repository.refresh_community_member_count(
                    self._user_id, keep
                )
                await self._graph_repository.refresh_community_member_count(
                    self._user_id, dissolve
                )
        return len(merged_into)

    async def _update_metadata(self, community_ids: list[str]) -> int:
        """生成并写入社区名称/摘要。"""
        enhanced = 0
        for community_id in community_ids:
            members = await self._graph_repository.community_members(
                self._user_id, community_id
            )
            if not members:
                continue
            relationships = await self._graph_repository.community_relationships(
                self._user_id, community_id
            )
            try:
                name, summary = await self._summarize(members, relationships)
                await self._graph_repository.update_community_metadata(
                    self._user_id, community_id, name, summary
                )
                enhanced += 1
            except Exception as e:
                logger.warning(
                    "记忆社区元数据生成失败，跳过 community=%s: %s", community_id, e
                )
        return enhanced

    async def _summarize(
        self,
        members: list[CommunityVoteEntity | CommunityMember],
        relationships: list[CommunityRelationFact],
    ) -> tuple[str, str]:
        """优先使用 LLM 摘要，缺失或空结果时兜底成员名。"""
        if self._summarizer:
            name, summary = await self._summarizer.summarize(
                members[: self._settings.memory_community_metadata_member_limit],
                relationships,
            )
            if name and summary:
                return name, summary
        return _fallback_metadata(members)

    def _weighted_vote(
        self,
        neighbors: list[CommunityVoteNeighbor],
        embedding: list[float],
        labels: dict[str, str],
    ) -> str | None:
        """按语义相似度和关系连接强度为邻居社区投票。"""
        votes: dict[str, float] = {}
        for neighbor in neighbors:
            label = labels.get(neighbor.id, neighbor.community_id)
            if not label:
                continue
            similarity = cosine_similarity(embedding, neighbor.embedding)
            weight = (
                self._settings.memory_community_semantic_weight * similarity
                + self._settings.memory_community_relation_weight
            )
            votes[str(label)] = votes.get(str(label), 0.0) + weight
        return max(votes, key=votes.__getitem__) if votes else None

    def _community_id_for_label(self, label: str) -> str:
        """已有社区标签保持不变，实体标签转成稳定社区 ID。"""
        if label.startswith("community-"):
            return label
        return stable_memory_graph_id(self._user_id, "community", label)


def _member_id(member: CommunityVoteEntity | CommunityMember) -> str:
    """兼容聚类投票实体和 API 成员实体的 id 字段。"""
    return getattr(member, "id", None) or getattr(member, "entity_id")


def _fallback_metadata(
    members: list[CommunityVoteEntity | CommunityMember],
) -> tuple[str, str]:
    """无 LLM 时使用成员名生成稳定兜底名称和摘要。"""
    names = [
        str(getattr(member, "name", None) or getattr(member, "entity_name", "")).strip()
        for member in members
    ]
    names = [name for name in names if name]
    if not names:
        return "未命名社区", "包含实体："
    return "、".join(names[:3]), f"包含实体：{', '.join(names[:10])}"
