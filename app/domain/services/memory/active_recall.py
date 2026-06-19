import asyncio
import logging

from app.domain.external.embedding import EmbeddingModel
from app.domain.models.memory_graph import (
    GraphRelationFact,
    MemoryActiveRecallCommunityHit,
    MemoryActiveRecallEventHit,
    MemoryGraphSearchHit,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from core.config import get_settings

logger = logging.getLogger(__name__)


class MemoryActiveRecall:
    """为 Agent 主链路主动召回相关记忆背景。"""

    def __init__(
        self,
        user_id: str,
        graph_repository: MemoryGraphRepository,
        embedding: EmbeddingModel,
    ) -> None:
        self._user_id = user_id
        self._graph_repository = graph_repository
        self._embedding = embedding
        self._settings = get_settings()

    async def recall_context(self, query: str) -> str:
        """召回 Insight 和实体事实，失败或超时返回空字符串。"""
        query = query.strip()
        if not query:
            return ""
        try:
            return await asyncio.wait_for(
                self._do_recall(query),
                timeout=self._settings.memory_active_recall_timeout_seconds,
            )
        except Exception as e:
            logger.warning("记忆主动召回失败，已跳过: %s", e)
            return ""

    async def _do_recall(self, query: str) -> str:
        """执行实际召回流程，一次 query embedding 复用给多路检索。"""
        query_embedding = await self._embedding.embed_one(query)
        lanes = [
            (
                "insights",
                self._graph_repository.search_insights_by_vector(
                    self._user_id,
                    query_embedding,
                    self._settings.memory_active_recall_insight_top_k,
                ),
            ),
            (
                "entities",
                self._graph_repository.search(
                    user_id=self._user_id,
                    query=query,
                    top_k=self._settings.memory_active_recall_entity_top_k,
                    query_embedding=query_embedding,
                ),
            ),
        ]
        if self._settings.memory_active_recall_include_communities:
            lanes.append(
                (
                    "communities",
                    self._graph_repository.search_communities_by_vector(
                        self._user_id,
                        query_embedding,
                        self._settings.memory_active_recall_community_top_k,
                    ),
                )
            )
        if self._settings.memory_active_recall_include_events:
            lanes.append(
                (
                    "events",
                    self._graph_repository.search_events_by_vector_or_text(
                        self._user_id,
                        query,
                        query_embedding,
                        self._settings.memory_active_recall_event_top_k,
                    ),
                )
            )
        lane_results = await asyncio.gather(
            *(lane for _, lane in lanes),
            return_exceptions=True,
        )
        results_by_name: dict[str, list] = {}
        for (name, _), result in zip(lanes, lane_results):
            if isinstance(result, Exception):
                logger.warning("记忆主动召回 %s 路查询失败，已跳过: %s", name, result)
                results_by_name[name] = []
            else:
                results_by_name[name] = list(result)
        insights = results_by_name.get("insights", [])
        entities = results_by_name.get("entities", [])
        communities = results_by_name.get("communities", [])
        events = results_by_name.get("events", [])
        min_score = self._settings.memory_active_recall_min_score
        insights = [insight for insight in insights if insight.score >= min_score]
        entities = [entity for entity in entities if entity.score >= min_score]
        communities = [
            community for community in communities if community.score >= min_score
        ]
        events = [event for event in events if event.score >= min_score]
        if not insights and not entities and not communities and not events:
            return ""
        parts = ["【关于用户的已知信息（供参考，可自然融入回答，不必刻意提及）】"]
        if insights:
            parts.append("我对用户的理解：" + "；".join(item.content for item in insights))
        if entities:
            parts.append("相关记忆：")
            for entity in entities:
                parts.extend(_format_entity_lines(entity))
        if communities:
            parts.append("相关主题社区：" + "；".join(_format_community(item) for item in communities))
        if events:
            parts.append("相关经历：" + "；".join(_format_event(item) for item in events))
        block = "\n".join(parts)
        max_chars = self._settings.memory_active_recall_max_chars
        return block if len(block) <= max_chars else block[:max_chars] + "..."


def _format_entity_lines(result: MemoryGraphSearchHit) -> list[str]:
    """把实体命中和一跳关系格式化为简短背景行。"""
    lines = [
        f"- {result.entity_name}：{result.description}"
        if result.description
        else f"- {result.entity_name}"
    ]
    for relation in result.relations[:2]:
        lines.append(f"  · {_format_relation(result, relation)}")
    return lines


def _format_relation(result: MemoryGraphSearchHit, relation: GraphRelationFact) -> str:
    """按关系方向还原事实读法。"""
    if relation.direction == "incoming":
        return f"{relation.neighbor_name} {relation.name} {result.entity_name}"
    return f"{result.entity_name} {relation.name} {relation.neighbor_name}"


def _format_community(result: MemoryActiveRecallCommunityHit) -> str:
    """格式化主动召回命中的主题社区。"""
    return f"{result.name}：{result.summary}" if result.summary else result.name


def _format_event(result: MemoryActiveRecallEventHit) -> str:
    """格式化主动召回命中的经历事件。"""
    event_time = result.event_time.isoformat() if result.event_time else "时间未明"
    participants = "、".join(participant.name for participant in result.participants)
    suffix = f" - {participants}" if participants else ""
    return f"{event_time} {result.title}{suffix}"
