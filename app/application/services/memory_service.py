from typing import Callable

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.domain.external.embedding import EmbeddingModel
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.task_dispatcher import TaskDispatcher
from app.domain.models.long_term_memory import LongTermMemory, MemorySource
from app.domain.models.memory_graph import MemoryConsolidationStats
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.repositories.vow import IUnitOfWork
from app.domain.services.memory import (
    LongTermMemoryManager,
    MemoryConsolidator,
    MemoryProfileSummarizer,
)


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
        self._user_id = user_id
        self._graph_repository = graph_repository
        self._profile_summarizer = (
            MemoryProfileSummarizer(llm=llm, json_parser=json_parser)
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
