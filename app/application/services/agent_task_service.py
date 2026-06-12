from typing import Type

from app.domain.external.task import Task


class AgentTaskService:
    """Agent任务生命周期服务"""

    def __init__(self, task_cls: Type[Task]) -> None:
        self._task_cls = task_cls

    async def shutdown(self) -> None:
        """关闭所有Agent任务资源"""
        await self._task_cls.destroy()
