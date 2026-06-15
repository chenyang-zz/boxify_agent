import logging

from app.domain.external.embedding import EmbeddingModel
from app.domain.models.memory_graph import EntityNode, MemoryReflectStats
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.services.memory.insight_generator import MemoryInsightGenerator
from core.config import get_settings

logger = logging.getLogger(__name__)


class MemoryReflector:
    """从长期图谱记忆中归纳高层 Insight。"""

    def __init__(
        self,
        user_id: str,
        graph_repository: MemoryGraphRepository,
        insight_generator: MemoryInsightGenerator | None = None,
        embedding: EmbeddingModel | None = None,
    ) -> None:
        self._user_id = user_id
        self._graph_repository = graph_repository
        self._insight_generator = insight_generator
        self._embedding = embedding
        self._settings = get_settings()

    async def reflect(self) -> MemoryReflectStats:
        """执行一次反思，按主题 upsert 洞察。"""
        if not self._insight_generator:
            return MemoryReflectStats(insights=0, skipped="no_llm")
        entities = await self._graph_repository.reflection_top_entities(
            self._user_id, self._settings.memory_reflection_top_k
        )
        if len(entities) < self._settings.memory_reflection_min_entities:
            return MemoryReflectStats(insights=0, skipped="too_few_entities")

        memory_block, name_to_id = await self._build_memory_block(entities)
        try:
            insights = await self._insight_generator.generate(
                memory_block=memory_block,
                min_insights=self._settings.memory_reflection_min_insights,
                max_insights=self._settings.memory_reflection_max_insights,
            )
        except Exception as e:
            logger.warning("记忆洞察反思失败: %s", e)
            return MemoryReflectStats(insights=0, error=str(e))
        if not insights:
            return MemoryReflectStats(insights=0)

        contents = [insight.content.strip() for insight in insights]
        embeddings = await self._embed_insights(contents)
        saved = 0
        for index, insight in enumerate(insights):
            theme = insight.theme.strip()
            content = insight.content.strip()
            if not theme or not content:
                continue
            if len(content) > 200:
                content = content[:200].rstrip() + "..."
            entity_ids = [
                name_to_id[name.strip()]
                for name in insight.based_on
                if name.strip() in name_to_id
            ]
            try:
                await self._graph_repository.upsert_insight(
                    user_id=self._user_id,
                    theme=theme,
                    content=content,
                    embedding=embeddings[index] if index < len(embeddings) else None,
                    importance=_clamp(insight.importance, default=0.6),
                    confidence=_clamp(insight.confidence, default=0.7),
                    source_count=len(entity_ids),
                    entity_ids=entity_ids,
                )
                saved += 1
            except Exception as e:
                logger.warning("记忆洞察写入失败，跳过 theme=%s: %s", theme, e)
        return MemoryReflectStats(insights=saved)

    async def _build_memory_block(
        self, entities: list[EntityNode]
    ) -> tuple[str, dict[str, str]]:
        """拼出给 LLM 的记忆清单，并返回实体名到 id 映射。"""
        lines: list[str] = []
        name_to_id: dict[str, str] = {}
        for entity in entities:
            name = entity.name.strip()
            if not name:
                continue
            name_to_id[name] = entity.id
            header = f"- 【{entity.type}】{name}"
            if entity.description:
                header += f"：{entity.description}"
            lines.append(header)
            if entity.core_facts:
                lines.append(f"    核心事实：{'；'.join(entity.core_facts[:5])}")
            if entity.traits:
                lines.append(f"    特质：{'、'.join(entity.traits[:5])}")
            statements = await self._graph_repository.reflection_entity_statements(
                self._user_id,
                entity.id,
                self._settings.memory_reflection_stmt_per_entity,
            )
            for statement in statements:
                lines.append(f"    · {statement}")
        return "\n".join(lines), name_to_id

    async def _embed_insights(self, contents: list[str]) -> list[list[float]]:
        """洞察向量化失败时返回空列表，仍允许写入洞察文本。"""
        if not self._embedding or not contents:
            return []
        try:
            return await self._embedding.embed(contents)
        except Exception as e:
            logger.warning("记忆洞察向量化失败，将仅写入文本: %s", e)
            return []


def _clamp(value: float, default: float) -> float:
    """把 LLM 分数夹到 0 到 1。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
