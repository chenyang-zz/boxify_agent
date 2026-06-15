from typing import Protocol


class TaskDispatcher(Protocol):
    """知识库后台任务派发协议"""

    async def dispatch_parse_document(self, document_id: str) -> None:
        """派发文档解析任务，调用方不关心具体队列实现。"""
        ...

    async def dispatch_extract_memory(self, memory_id: str) -> None:
        """派发长期记忆图谱萃取任务。"""
        ...

    async def dispatch_consolidate_memory(self, user_id: str) -> None:
        """派发当前用户长期记忆巩固任务。"""
        ...
