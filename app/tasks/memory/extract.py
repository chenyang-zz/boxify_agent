import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from app.bootstrap.memory import build_memory_graph_extractor_for_user
from app.celery_app import celery_app
from app.domain.repositories.vow import IUnitOfWork
from app.domain.services.memory.graph_extractor import MemoryGraphExtractor
from app.infrastructure.storage.neo4j import get_neo4j
from app.infrastructure.storage.postgres import get_postgres, get_uow

logger = logging.getLogger(__name__)


async def run_extract_memory(
    memory_id: str,
    uow_factory: Callable[[], IUnitOfWork],
    pipeline_factory: Callable[
        [str], MemoryGraphExtractor | Awaitable[MemoryGraphExtractor]
    ],
) -> None:
    """执行单条长期记忆图谱萃取并回写 PG 状态。"""
    memory = None
    async with uow_factory() as uow:
        for_user = await _find_memory_user(uow, memory_id)
        if not for_user:
            return
        memory = await uow.memory.get_by_user(for_user, memory_id)
        if not memory:
            return
        memory.mark_extracting()
        await uow.memory.save(memory)

    try:
        pipeline = pipeline_factory(memory.user_id)
        if inspect.isawaitable(pipeline):
            pipeline = await pipeline
        stats = await pipeline.extract_memory(
            memory_id=memory.id,
            user_id=memory.user_id,
            content=memory.content,
        )
        async with uow_factory() as uow:
            latest = await uow.memory.get_by_user(memory.user_id, memory.id)
            if not latest:
                return
            latest.mark_completed(
                summary=latest.summary,
                keywords=latest.keywords,
                graph_dialogue_id=stats.dialogue_id,
                graph_stats=stats,
            )
            await uow.memory.save(latest)
    except Exception as e:
        logger.error("长期记忆图谱萃取失败: %s", e, exc_info=True)
        async with uow_factory() as uow:
            latest = await uow.memory.get_by_user(memory.user_id, memory.id)
            if latest:
                latest.mark_failed(str(e))
                await uow.memory.save(latest)


async def _run(memory_id: str) -> None:
    """初始化 worker 外部资源并执行单条记忆萃取任务。"""
    await get_postgres().init()
    await get_neo4j().init()
    try:
        await run_extract_memory(
            memory_id=memory_id,
            uow_factory=get_uow,
            pipeline_factory=build_memory_graph_extractor_for_user,
        )
    finally:
        await get_neo4j().shutdown()
        await get_postgres().shutdown()


async def _find_memory_user(uow: IUnitOfWork, memory_id: str) -> str | None:
    """当前仓储接口按用户取 memory，这里用受控内部查询兼容测试 UoW。"""
    finder = getattr(uow.memory, "get_user_id_by_memory_id", None)
    if finder:
        return await finder(memory_id)
    saved = getattr(uow.memory, "saved", [])
    for memory in saved:
        if memory.id == memory_id:
            return memory.user_id
    return None


@celery_app.task(name="app.tasks.memory_extract_memory")
def extract_memory_task(memory_id: str) -> str:
    """Celery 任务入口。"""
    asyncio.run(_run(memory_id))
    return memory_id
