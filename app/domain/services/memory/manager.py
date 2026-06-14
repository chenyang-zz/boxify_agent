import logging
from typing import Callable, Protocol

from app.domain.external.embedding import EmbeddingModel
from app.domain.external.task_dispatcher import TaskDispatcher
from app.domain.models.long_term_memory import (
    LongTermMemory,
    MemorySource,
    MemoryStatus,
)
from app.domain.models.memory_graph import (
    GraphRelationFact,
    LongTermMemoryGraphData,
    MemoryGraphResult,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.repositories.vow import IUnitOfWork

logger = logging.getLogger(__name__)


class MemorySearch(Protocol):
    """长期记忆检索能力。"""

    async def search(self, query: str, top_k: int) -> list[LongTermMemory]:
        """检索长期记忆。"""
        ...


class LongTermMemoryManager:
    """当前用户长期记忆管理器。"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        user_id: str,
        task_dispatcher: TaskDispatcher | None = None,
        graph_repository: MemoryGraphRepository | None = None,
        embedding: EmbeddingModel | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._user_id = user_id
        self._task_dispatcher = task_dispatcher
        self._graph_repository = graph_repository
        self._embedding = embedding

    async def remember_text(
        self,
        content: str,
        source: MemorySource,
        source_session_id: str | None = None,
    ) -> LongTermMemory:
        """保存一条当前用户长期记忆。"""
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空")

        memory = LongTermMemory(
            user_id=self._user_id,
            content=content,
            source=source,
            source_session_id=source_session_id,
            summary=content,
            keywords=self.extract_keywords(content),
            status=MemoryStatus.PENDING,
        )
        async with self._uow_factory() as uow:
            await uow.memory.save(memory)
        if self._task_dispatcher:
            await self._task_dispatcher.dispatch_extract_memory(memory.id)
        return memory

    async def list_memories(
        self, page: int, page_size: int
    ) -> tuple[list[LongTermMemory], int]:
        """分页读取当前用户记忆。"""
        async with self._uow_factory() as uow:
            return await uow.memory.list_by_user(self._user_id, page, page_size)

    async def search(self, query: str, top_k: int) -> list[LongTermMemory]:
        """检索当前用户长期记忆。"""
        query = query.strip()
        if not query:
            raise ValueError("检索关键词不能为空")
        top_k = max(1, min(top_k, 20))
        graph_results = await self._search_graph(query, top_k)
        if graph_results:
            return graph_results
        async with self._uow_factory() as uow:
            return await uow.memory.search_by_user(self._user_id, query, top_k)

    async def delete_memory(self, memory_id: str) -> bool:
        """删除当前用户记忆，返回是否删除成功。"""
        async with self._uow_factory() as uow:
            return await uow.memory.delete_by_user(self._user_id, memory_id)

    @classmethod
    def extract_keywords(cls, content: str) -> list[str]:
        """提取轻量关键词，给 V1 检索和响应展示使用。"""
        normalized = content.replace("，", " ").replace("。", " ").replace(",", " ")
        return [word for word in normalized.split() if word][:8]

    async def _search_graph(self, query: str, top_k: int) -> list[LongTermMemory]:
        if not self._graph_repository:
            return []
        query_embedding = None
        if self._embedding:
            try:
                query_embedding = await self._embedding.embed_one(query)
            except Exception as e:
                logger.warning("记忆图谱查询向量生成失败，仅使用全文检索: %s", e)
        try:
            results = await self._graph_repository.search(
                user_id=self._user_id,
                query=query,
                top_k=top_k,
                query_embedding=query_embedding,
            )
        except Exception as e:
            logger.warning("记忆图谱检索失败，降级到 PG 检索: %s", e)
            return []
        return [self._graph_result_to_memory(result) for result in results]

    def _graph_result_to_memory(self, result: MemoryGraphResult) -> LongTermMemory:
        relation_facts = result.relations
        summary = result.source_memory_summary or result.description or result.entity_name
        content = self._format_graph_content(result, relation_facts)
        return LongTermMemory(
            id=result.source_memory_id or result.entity_id,
            user_id=self._user_id,
            content=content,
            status=MemoryStatus.COMPLETED,
            summary=summary,
            keywords=[value for value in [result.entity_name, result.entity_type] if value],
            graph_data=LongTermMemoryGraphData.from_result(result),
        )

    @staticmethod
    def _format_graph_content(
        result: MemoryGraphResult, relation_facts: list[GraphRelationFact]
    ) -> str:
        if result.source_memory_summary:
            return result.source_memory_summary
        if relation_facts:
            facts = [
                (
                    f"{fact.neighbor_name} "
                    f"{fact.name} "
                    f"{result.entity_name}"
                    if fact.direction == "incoming"
                    else (
                        f"{result.entity_name} "
                        f"{fact.name} "
                        f"{fact.neighbor_name}"
                    )
                )
                for fact in relation_facts
            ]
            return "；".join(facts)
        return result.description or result.entity_name
