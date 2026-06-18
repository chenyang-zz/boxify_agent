import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.celery_app import celery_app
from app.infrastructure.storage.postgres import get_postgres, get_uow

logger = logging.getLogger(__name__)

ScheduledStats = dict[str, int]


async def run_scheduled_consolidate_memory(
    uow_factory=get_uow,
    dispatch: Callable[[str], Awaitable[None]] | None = None,
) -> ScheduledStats:
    return await _run_for_active_users(
        uow_factory=uow_factory,
        dispatch=dispatch or _dispatch_consolidate_memory,
        task_name="consolidate",
    )


async def run_scheduled_cluster_memory(
    uow_factory=get_uow,
    dispatch: Callable[[str, str | None], Awaitable[None]] | None = None,
) -> ScheduledStats:
    async def dispatch_full_cluster(user_id: str) -> None:
        if dispatch:
            await dispatch(user_id, None)
            return
        await _dispatch_cluster_memory(user_id, None)

    return await _run_for_active_users(
        uow_factory=uow_factory,
        dispatch=dispatch_full_cluster,
        task_name="cluster",
    )


async def run_scheduled_reflect_memory(
    uow_factory=get_uow,
    dispatch: Callable[[str], Awaitable[None]] | None = None,
) -> ScheduledStats:
    return await _run_for_active_users(
        uow_factory=uow_factory,
        dispatch=dispatch or _dispatch_reflect_memory,
        task_name="reflect",
    )


async def _run_for_active_users(
    uow_factory,
    dispatch: Callable[[str], Awaitable[None]],
    task_name: str,
) -> ScheduledStats:
    async with uow_factory() as uow:
        user_ids = await uow.user.list_active_ids()

    stats = {
        "users_scanned": len(user_ids),
        "dispatched": 0,
        "skipped": 0,
        "failed": 0,
    }
    for user_id in user_ids:
        try:
            await dispatch(user_id)
            stats["dispatched"] += 1
        except Exception as e:
            stats["failed"] += 1
            logger.warning(
                "定时记忆任务派发失败: task=%s user_id=%s error=%s",
                task_name,
                user_id,
                e,
            )
    return stats


async def _dispatch_consolidate_memory(user_id: str) -> None:
    from app.tasks.memory.consolidate import consolidate_memory_task

    consolidate_memory_task.delay(user_id)


async def _dispatch_cluster_memory(user_id: str, dialogue_id: str | None) -> None:
    from app.tasks.memory.cluster import cluster_memory_task

    cluster_memory_task.delay(user_id, dialogue_id)


async def _dispatch_reflect_memory(user_id: str) -> None:
    from app.tasks.memory.reflect import reflect_memory_task

    reflect_memory_task.delay(user_id)


async def _run_with_postgres(
    runner: Callable[[], Awaitable[ScheduledStats]],
) -> ScheduledStats:
    await get_postgres().init()
    try:
        return await runner()
    finally:
        await get_postgres().shutdown()


@celery_app.task(name="app.tasks.memory_scheduled_consolidate")
def scheduled_consolidate_memory_task() -> ScheduledStats:
    return asyncio.run(_run_with_postgres(run_scheduled_consolidate_memory))


@celery_app.task(name="app.tasks.memory_scheduled_cluster")
def scheduled_cluster_memory_task() -> ScheduledStats:
    return asyncio.run(_run_with_postgres(run_scheduled_cluster_memory))


@celery_app.task(name="app.tasks.memory_scheduled_reflect")
def scheduled_reflect_memory_task() -> ScheduledStats:
    return asyncio.run(_run_with_postgres(run_scheduled_reflect_memory))
