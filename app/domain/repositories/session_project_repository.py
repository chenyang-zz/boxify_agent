from abc import ABC, abstractmethod

from app.domain.models.session_project import SessionProject


class SessionProjectRepository(ABC):
    """会话项目仓储接口，项目始终按用户维度隔离。"""

    @abstractmethod
    async def save(self, project: SessionProject) -> None:
        """新增或更新项目。"""
        ...

    @abstractmethod
    async def list_by_user(self, user_id: str) -> list[SessionProject]:
        """查询当前用户项目列表。"""
        ...

    @abstractmethod
    async def get_by_id_for_user(
        self, project_id: str, user_id: str
    ) -> SessionProject | None:
        """按用户维度查询项目。"""
        ...

    @abstractmethod
    async def delete_by_id_for_user(self, project_id: str, user_id: str) -> None:
        """删除当前用户的项目。"""
        ...
