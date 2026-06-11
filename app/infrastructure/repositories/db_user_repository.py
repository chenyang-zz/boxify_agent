from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.user import User
from app.domain.repositories.user_repository import UserRepository
from app.infrastructure.models.user import UserModel


class DBUserRepository(UserRepository):
    """基于数据库的用户仓储"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def get_by_id(self, user_id: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_username(self, username: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.username == username)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def count(self) -> int:
        stmt = select(func.count()).select_from(UserModel)
        result = await self.db_session.execute(stmt)
        return int(result.scalar_one())

    async def save(self, user: User) -> None:
        stmt = select(UserModel).where(UserModel.id == user.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            self.db_session.add(UserModel.from_domain(user))
            return
        record.update_from_domain(user)
