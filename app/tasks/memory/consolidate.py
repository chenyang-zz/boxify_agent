import asyncio
from collections.abc import Awaitable, Callable

from app.bootstrap.memory import build_memory_consolidation_service_for_user
from app.celery_app import celery_app
from app.domain.models.memory_graph import MemoryConsolidationStats
from app.domain.services.memory.consolidator import MemoryConsolidator


async def run_consolidate_memory(
    user_id: str,
    service_factory: Callable[
        [str], Awaitable[MemoryConsolidator]
    ] = build_memory_consolidation_service_for_user,
) -> MemoryConsolidationStats:
    """执行单个用户记忆巩固，便于 Celery 任务和测试复用。"""
    service = await service_factory(user_id)
    return await service.consolidate()


async def _run(user_id: str) -> MemoryConsolidationStats:
    """Celery 内部异步入口，隔离同步任务包装和领域执行逻辑。"""
    return await run_consolidate_memory(user_id)


@celery_app.task(name="app.tasks.memory_consolidate_memory")
def consolidate_memory_task(user_id: str) -> dict[str, int]:
    """Celery 任务入口。"""
    return asyncio.run(_run(user_id)).model_dump()
