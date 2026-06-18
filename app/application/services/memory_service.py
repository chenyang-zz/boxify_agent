import logging
from typing import Callable

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.domain.external.embedding import EmbeddingModel
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.task_dispatcher import TaskDispatcher
from app.domain.models.long_term_memory import (
    LongTermMemory,
    MemoryDetailResult,
    MemorySource,
)
from app.domain.models.memory_graph import (
    MEMORY_QUALITY_ISSUE_CATEGORIES,
    MemoryConsolidationStats,
    MemoryReflectStats,
)
from app.domain.models.memory_graph import (
    CommunityMemberResult,
    CommunityRelationResult,
    CommunityResult,
    InsightResult,
    MemoryEntitySubgraphResult,
    MemoryCommunityClusterStats,
    MemoryGraphViewResult,
    MemoryMergeDuplicatesResult,
    MemoryProfileGroupResult,
    MemoryProfileResult,
    MemoryQualityGraphCountsResult,
    MemoryQualityIssueListResult,
    MemoryQualityIssueSummaryResult,
    MemoryQualityOverviewResult,
    MemoryRelationHistoryResult,
    MemoryTimelineEventResult,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.repositories.vow import IUnitOfWork
from app.domain.services.memory import (
    LongTermMemoryManager,
    MemoryCommunityClusterer,
    MemoryCommunitySummarizer,
    MemoryConsolidator,
    MemoryInsightGenerator,
    MemoryProfileSummarizer,
    MemoryReflector,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """长期记忆应用服务。"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        user_id: str,
        task_dispatcher: TaskDispatcher | None = None,
        graph_repository: MemoryGraphRepository | None = None,
        embedding: EmbeddingModel | None = None,
        llm: LLM | None = None,
        json_parser: JSONParser | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._user_id = user_id
        self._graph_repository = graph_repository
        self._embedding = embedding
        self._profile_summarizer = (
            MemoryProfileSummarizer(llm=llm, json_parser=json_parser)
            if llm and json_parser
            else None
        )
        self._insight_generator = (
            MemoryInsightGenerator(llm=llm, json_parser=json_parser)
            if llm and json_parser
            else None
        )
        self._community_summarizer = (
            MemoryCommunitySummarizer(llm=llm, json_parser=json_parser)
            if llm and json_parser
            else None
        )
        self._memory = LongTermMemoryManager(
            uow_factory=uow_factory,
            user_id=user_id,
            task_dispatcher=task_dispatcher,
            graph_repository=graph_repository,
            embedding=embedding,
        )

    async def remember_text(self, content: str) -> LongTermMemory:
        """主动记住一段文本，并异步萃取图谱。"""
        try:
            return await self._memory.remember_text(content, source=MemorySource.MANUAL)
        except ValueError as e:
            raise BadRequestError(str(e)) from e

    async def remember_session_text(
        self, content: str, session_id: str
    ) -> LongTermMemory:
        """从会话消息沉淀长期记忆。"""
        try:
            return await self._memory.remember_text(
                content,
                source=MemorySource.SESSION,
                source_session_id=session_id,
            )
        except ValueError as e:
            raise BadRequestError(str(e)) from e

    async def list_memories(
        self, page: int, page_size: int
    ) -> tuple[list[LongTermMemory], int]:
        """分页读取当前用户记忆。"""
        return await self._memory.list_memories(page, page_size)

    async def search(self, query: str, top_k: int) -> list[LongTermMemory]:
        """检索当前用户记忆。"""
        try:
            return await self._memory.search(query, top_k)
        except ValueError as e:
            raise BadRequestError(str(e)) from e

    async def delete_memory(self, memory_id: str) -> None:
        """删除当前用户记忆。"""
        if not await self._memory.delete_memory(memory_id):
            raise NotFoundError("记忆不存在或无权访问")

    async def detail(self, memory_id: str) -> MemoryDetailResult:
        """读取当前用户单条长期记忆及其图谱溯源详情。"""
        async with self._uow_factory() as uow:
            memory = await uow.memory.get_by_user(self._user_id, memory_id)
        if memory is None:
            raise NotFoundError("记忆不存在或无权访问")
        if not self._graph_repository:
            return MemoryDetailResult(
                **memory.model_dump(mode="python"),
                graph_available=False,
                trace=None,
            )
        try:
            trace = await self._graph_repository.memory_trace(
                self._user_id,
                memory_id,
            )
        except Exception as e:
            logger.warning("记忆图谱溯源查询失败，返回 PG 详情: %s", e)
            return MemoryDetailResult(
                **memory.model_dump(mode="python"),
                graph_available=False,
                trace=None,
            )
        return MemoryDetailResult(
            **memory.model_dump(mode="python"),
            graph_available=True,
            trace=trace,
        )

    async def consolidate(self) -> MemoryConsolidationStats:
        """手动执行当前用户记忆巩固。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法执行巩固")
        service = MemoryConsolidator(
            user_id=self._user_id,
            graph_repository=self._graph_repository,
            profile_summarizer=self._profile_summarizer,
        )
        return await service.consolidate()

    async def reflect(self) -> MemoryReflectStats:
        """手动执行当前用户记忆反思。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法执行反思")
        reflector = MemoryReflector(
            user_id=self._user_id,
            graph_repository=self._graph_repository,
            insight_generator=self._insight_generator,
            embedding=self._embedding,
        )
        return await reflector.reflect()

    async def cluster(self) -> MemoryCommunityClusterStats:
        """手动执行当前用户记忆社区聚类。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法执行社区聚类")
        clusterer = MemoryCommunityClusterer(
            user_id=self._user_id,
            graph_repository=self._graph_repository,
            summarizer=self._community_summarizer,
        )
        return await clusterer.cluster()

    async def list_communities(self) -> list[CommunityResult]:
        """列出当前用户记忆社区。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询社区")
        return await self._graph_repository.list_communities(self._user_id)

    async def community_detail(
        self, community_id: str
    ) -> tuple[list[CommunityMemberResult], list[CommunityRelationResult]]:
        """读取当前用户指定社区成员和社区内关系。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询社区")
        members = await self._graph_repository.community_members(
            self._user_id, community_id
        )
        relationships = await self._graph_repository.community_relationships(
            self._user_id, community_id
        )
        return members, relationships

    async def timeline(self, limit: int = 50) -> list[MemoryTimelineEventResult]:
        """读取当前用户记忆事件时间线。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询事件时间线")
        limit = max(1, min(limit, 200))
        return await self._graph_repository.event_timeline(self._user_id, limit)

    async def graph(self) -> MemoryGraphViewResult:
        """读取当前用户完整实体关系图。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询实体关系图")
        nodes = await self._graph_repository.graph_nodes(self._user_id)
        edges = await self._graph_repository.graph_edges(self._user_id)
        communities = await self._graph_repository.list_communities(self._user_id)
        return MemoryGraphViewResult(
            nodes=nodes,
            edges=edges,
            communities=communities,
        )

    async def entity_subgraph(self, entity_id: str) -> MemoryEntitySubgraphResult:
        """读取当前用户单实体一跳子图。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询实体关系图")
        subgraph = await self._graph_repository.entity_subgraph(
            self._user_id, entity_id
        )
        if not any(node.id == entity_id for node in subgraph.nodes):
            raise NotFoundError("实体不存在或无权访问")
        return subgraph

    async def profile(self) -> MemoryProfileResult:
        """读取当前用户记忆画像。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询画像")
        entities = await self._graph_repository.profile_entities(self._user_id)
        type_counts = await self._graph_repository.entity_type_counts(self._user_id)
        grouped: dict[str, list] = {}
        for entity in entities:
            grouped.setdefault(entity.type, []).append(entity)
        return MemoryProfileResult(
            total=len(entities),
            type_counts=type_counts,
            groups=[
                MemoryProfileGroupResult(type=entity_type, entities=items)
                for entity_type, items in grouped.items()
            ],
        )

    async def list_insights(self) -> list[InsightResult]:
        """列出当前用户长期记忆洞察。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询洞察")
        return await self._graph_repository.list_insights(self._user_id)

    async def delete_insight(self, insight_id: str) -> None:
        """删除当前用户单条洞察。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法删除洞察")
        if not await self._graph_repository.delete_insight(self._user_id, insight_id):
            raise NotFoundError("洞察不存在或无权访问")

    async def delete_entity(self, entity_id: str) -> None:
        """删除当前用户单个图谱实体。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法删除实体")
        if not await self._graph_repository.delete_entity(self._user_id, entity_id):
            raise NotFoundError("实体不存在或无权访问")

    async def merge_duplicates(self) -> MemoryMergeDuplicatesResult:
        """合并当前用户历史同名同类型重复实体。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法合并重复实体")
        return await self._graph_repository.merge_duplicate_entities(self._user_id)

    async def relation_history(
        self, entity_id: str, predicate: str | None = None
    ) -> list[MemoryRelationHistoryResult]:
        """读取当前用户单实体一跳关系历史。"""
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询关系历史")
        relations = await self._graph_repository.entity_relation_history(
            self._user_id,
            entity_id,
            predicate=predicate,
        )
        if relations is None:
            raise NotFoundError("实体不存在或无权访问")
        return relations

    async def quality(self) -> MemoryQualityOverviewResult:
        """读取当前用户记忆质量审计总览。"""
        async with self._uow_factory() as uow:
            pg_status_counts = await uow.memory.status_counts(self._user_id)
            recent_failed = await uow.memory.recent_failed(self._user_id, limit=5)
        pg_total = sum(pg_status_counts.values())
        if not self._graph_repository:
            return MemoryQualityOverviewResult(
                pg_total=pg_total,
                pg_status_counts=pg_status_counts,
                recent_failed=recent_failed,
                graph_available=False,
            )
        try:
            graph_counts = await self._graph_repository.quality_graph_counts(
                self._user_id
            )
            issue_summary = await self._graph_repository.quality_issue_summary(
                self._user_id
            )
        except Exception as e:
            logger.warning("记忆质量审计图谱统计失败，返回 PG 统计: %s", e)
            return MemoryQualityOverviewResult(
                pg_total=pg_total,
                pg_status_counts=pg_status_counts,
                recent_failed=recent_failed,
                graph_available=False,
                graph_counts=MemoryQualityGraphCountsResult(),
                issue_summary=MemoryQualityIssueSummaryResult(),
            )
        return MemoryQualityOverviewResult(
            pg_total=pg_total,
            pg_status_counts=pg_status_counts,
            recent_failed=recent_failed,
            graph_available=True,
            graph_counts=graph_counts,
            issue_summary=issue_summary,
        )

    async def quality_issues(
        self, category: str, limit: int
    ) -> MemoryQualityIssueListResult:
        """读取指定类别的记忆质量问题样本。"""
        if category not in MEMORY_QUALITY_ISSUE_CATEGORIES:
            raise BadRequestError("未知质量问题类别")
        if not self._graph_repository:
            raise BadRequestError("记忆图谱不可用，无法查询质量问题")
        limit = max(1, min(limit, 200))
        try:
            return await self._graph_repository.quality_issues(
                self._user_id,
                category,
                limit,
            )
        except Exception as e:
            logger.warning("记忆质量问题查询失败: %s", e)
            raise BadRequestError("记忆图谱不可用，无法查询质量问题") from e
