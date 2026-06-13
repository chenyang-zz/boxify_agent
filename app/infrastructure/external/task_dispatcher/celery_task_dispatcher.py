from app.domain.external.task_dispatcher import TaskDispatcher


class CeleryTaskDispatcher(TaskDispatcher):
    """基于 Celery 的知识库任务派发器。"""

    async def dispatch_parse_document(self, document_id: str) -> None:
        """发送解析任务到 broker，FastAPI 请求线程不直接执行解析。"""
        from app.tasks.notebook.document_parse import parse_document_task

        parse_document_task.delay(document_id)
