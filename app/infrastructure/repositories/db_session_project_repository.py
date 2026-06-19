from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.session_project import SessionProject
from app.domain.repositories.session_project_repository import SessionProjectRepository
from app.infrastructure.models.session_project import SessionProjectModel


class DBSessionProjectRepository(SessionProjectRepository):
    """基于数据库的会话项目仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def save(self, project: SessionProject) -> None:
        """新增或更新项目。"""
        stmt = select(SessionProjectModel).where(SessionProjectModel.id == project.id)
        record = (await self.db_session.execute(stmt)).scalar_one_or_none()
        if not record:
            self.db_session.add(SessionProjectModel.from_domain(project))
            return
        record.update_from_domain(project)

    async def list_by_user(self, user_id: str) -> list[SessionProject]:
        """查询当前用户项目列表。"""
        stmt = (
            select(SessionProjectModel)
            .where(SessionProjectModel.user_id == user_id)
            .order_by(SessionProjectModel.sort_order, SessionProjectModel.created_at)
        )
        records = (await self.db_session.execute(stmt)).scalars().all()
        return [record.to_domain() for record in records]

    async def get_by_id_for_user(
        self, project_id: str, user_id: str
    ) -> SessionProject | None:
        """按用户维度查询项目。"""
        stmt = select(SessionProjectModel).where(
            SessionProjectModel.id == project_id,
            SessionProjectModel.user_id == user_id,
        )
        record = (await self.db_session.execute(stmt)).scalar_one_or_none()
        return record.to_domain() if record else None

    async def delete_by_id_for_user(self, project_id: str, user_id: str) -> None:
        """删除当前用户的项目。"""
        await self.db_session.execute(
            delete(SessionProjectModel).where(
                SessionProjectModel.id == project_id,
                SessionProjectModel.user_id == user_id,
            )
        )
