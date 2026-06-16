from app.domain.external.task_dispatcher import TaskDispatcher
from app.infrastructure.storage.redis import get_redis
from core.config import get_settings

_REFLECTION_TRIGGER_SCRIPT = """
local total = redis.call('INCRBY', KEYS[1], ARGV[1])
if total >= tonumber(ARGV[2]) then
    redis.call('DEL', KEYS[1])
    return 1
end
return 0
"""


class CeleryTaskDispatcher(TaskDispatcher):
    """基于 Celery 的知识库任务派发器。"""

    async def dispatch_parse_document(self, document_id: str) -> None:
        """发送解析任务到 broker，FastAPI 请求线程不直接执行解析。"""
        from app.tasks.notebook.document_parse import parse_document_task

        parse_document_task.delay(document_id)

    async def dispatch_extract_memory(self, memory_id: str) -> None:
        """发送长期记忆图谱萃取任务到 broker。"""
        from app.tasks.memory.extract import extract_memory_task

        extract_memory_task.delay(memory_id)

    async def dispatch_consolidate_memory(self, user_id: str) -> None:
        """发送长期记忆巩固任务到 broker。"""
        from app.tasks.memory.consolidate import consolidate_memory_task

        consolidate_memory_task.delay(user_id)

    async def dispatch_reflect_memory(self, user_id: str, entity_count: int) -> bool:
        """累计实体数，达到阈值时原子清零并发送长期记忆反思任务。"""
        if entity_count <= 0:
            return False
        settings = get_settings()
        key = f"memory:reflection:pending:{user_id}"
        should_dispatch = await get_redis().client.eval(
            _REFLECTION_TRIGGER_SCRIPT,
            1,
            key,
            entity_count,
            settings.memory_reflection_trigger_threshold,
        )
        if int(should_dispatch) != 1:
            return False

        from app.tasks.memory.reflect import reflect_memory_task

        reflect_memory_task.delay(user_id)
        return True

    async def dispatch_cluster_memory(self, user_id: str, dialogue_id: str) -> None:
        """发送长期记忆社区聚类任务到 broker。"""
        from app.tasks.memory.cluster import cluster_memory_task

        cluster_memory_task.delay(user_id, dialogue_id)
