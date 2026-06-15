import logging
from datetime import datetime, timedelta

from app.domain.models.memory_graph import (
    MemoryConsolidationStats,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.services.memory.profile_summarizer import MemoryProfileSummarizer
from core.config import get_settings

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """记忆动力学巩固器：短期提升长期，并为长期实体生成画像摘要。"""

    def __init__(
        self,
        user_id: str,
        graph_repository: MemoryGraphRepository,
        profile_summarizer: MemoryProfileSummarizer | None = None,
    ) -> None:
        self._user_id = user_id
        self._graph_repository = graph_repository
        self._profile_summarizer = profile_summarizer
        self._settings = get_settings()

    async def consolidate(self) -> MemoryConsolidationStats:
        """执行一次当前用户记忆巩固，返回可记录的统计。"""
        age_before = (
            datetime.now()
            - timedelta(hours=self._settings.memory_consolidate_min_age_hours)
        ).isoformat()
        stats = await self._graph_repository.promote_short_to_long(
            user_id=self._user_id,
            min_access=self._settings.memory_consolidate_min_access,
            min_importance=self._settings.memory_consolidate_min_importance,
            min_mention=self._settings.memory_consolidate_min_mention,
            age_before=age_before,
        )
        enhanced_profiles = 0
        if self._profile_summarizer:
            enhanced_profiles = await self._enhance_profiles()
        return MemoryConsolidationStats(
            promoted_entities=stats.promoted_entities,
            promoted_statements=stats.promoted_statements,
            enhanced_profiles=enhanced_profiles,
        )

    async def _enhance_profiles(self) -> int:
        """对 top-K 长期实体做画像增强，单实体失败只跳过。"""
        entities = await self._graph_repository.top_long_term_entities(
            self._user_id, self._settings.memory_consolidate_profile_top_k
        )
        enhanced = 0
        for entity in entities:
            try:
                statements = await self._graph_repository.entity_statements(
                    self._user_id, entity.id
                )
                if len(statements) < 2:
                    continue
                core_facts, traits = await self._profile_summarizer.summarize(
                    entity, statements
                )
                if not core_facts and not traits:
                    continue
                await self._graph_repository.write_entity_profile(
                    self._user_id, entity.id, core_facts, traits
                )
                enhanced += 1
            except Exception as e:
                logger.warning("长期记忆实体画像增强失败，跳过实体 %s: %s", entity, e)
        return enhanced
