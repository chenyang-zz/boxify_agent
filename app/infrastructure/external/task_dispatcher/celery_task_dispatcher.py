from app.domain.external.task_dispatcher import TaskDispatcher


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
