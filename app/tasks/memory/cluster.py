import asyncio
from collections.abc import Awaitable, Callable

from app.bootstrap.memory import build_memory_community_clusterer_for_user
from app.celery_app import celery_app
from app.domain.models.memory_graph import MemoryCommunityClusterStats
from app.domain.services.memory.community_clusterer import MemoryCommunityClusterer
from app.infrastructure.storage.neo4j import get_neo4j
from app.infrastructure.storage.postgres import get_postgres


async def run_cluster_memory(
    user_id: str,
    dialogue_id: str | None = None,
    clusterer_factory: Callable[
        [str], Awaitable[MemoryCommunityClusterer]
    ] = build_memory_community_clusterer_for_user,
) -> MemoryCommunityClusterStats:
    """执行单个用户记忆社区聚类，便于 Celery 任务和测试复用。"""
    clusterer = await clusterer_factory(user_id)
    return await clusterer.cluster(dialogue_id=dialogue_id)


async def _run(user_id: str, dialogue_id: str | None = None) -> MemoryCommunityClusterStats:
    """初始化 worker 外部资源并执行单个用户的社区聚类。"""
    await get_postgres().init()
    await get_neo4j().init()
    try:
        return await run_cluster_memory(user_id=user_id, dialogue_id=dialogue_id)
    finally:
        await get_neo4j().shutdown()
        await get_postgres().shutdown()


@celery_app.task(name="app.tasks.memory_cluster_memory")
def cluster_memory_task(user_id: str, dialogue_id: str | None = None) -> dict:
    """Celery 任务入口。"""
    return asyncio.run(_run(user_id, dialogue_id)).model_dump()
