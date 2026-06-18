import asyncio
from collections.abc import Awaitable, Callable

from app.bootstrap.memory import build_memory_reflector_for_user
from app.celery_app import celery_app
from app.domain.models.memory_graph import MemoryReflectStats
from app.domain.services.memory.reflector import MemoryReflector
from app.infrastructure.storage.neo4j import get_neo4j
from app.infrastructure.storage.postgres import get_postgres


async def run_reflect_memory(
    user_id: str,
    reflector_factory: Callable[
        [str], Awaitable[MemoryReflector]
    ] = build_memory_reflector_for_user,
) -> MemoryReflectStats:
    """执行单个用户记忆反思，便于 Celery 任务和测试复用。"""
    reflector = await reflector_factory(user_id)
    return await reflector.reflect()


async def _run(user_id: str) -> MemoryReflectStats:
    """初始化 worker 外部资源并执行单个用户的记忆反思。"""
    await get_postgres().init()
    await get_neo4j().init()
    try:
        return await run_reflect_memory(user_id)
    finally:
        await get_neo4j().shutdown()
        await get_postgres().shutdown()


@celery_app.task(name="app.tasks.memory_reflect_memory")
def reflect_memory_task(user_id: str) -> dict[str, int | str | None]:
    """Celery 任务入口。"""
    return asyncio.run(_run(user_id)).model_dump()
