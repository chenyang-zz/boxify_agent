from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.security import AppConfigEncryption
from app.domain.models.app_config import AppConfig, create_default_app_config
from app.domain.repositories.app_config_repository import AppConfigRepository
from app.infrastructure.models.app_config import AppConfigModel


class DBAppConfigRepository(AppConfigRepository):
    """基于数据库的用户应用配置仓储"""

    def __init__(
        self,
        db_session: AsyncSession,
        encryption: AppConfigEncryption | None = None,
        encryption_key: str = "",
    ) -> None:
        self.db_session = db_session
        self._encryption = encryption or AppConfigEncryption(encryption_key)

    async def get_by_user_id(self, user_id: str) -> Optional[AppConfig]:
        stmt = select(AppConfigModel).where(AppConfigModel.user_id == user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain(self._encryption) if record else None

    async def get_or_create_default(self, user_id: str) -> AppConfig:
        app_config = await self.get_by_user_id(user_id)
        if app_config:
            return app_config

        app_config = create_default_app_config()
        self.db_session.add(
            AppConfigModel.from_domain(
                user_id,
                app_config,
                self._encryption,
            )
        )
        await self.db_session.flush()
        return app_config

    async def save(self, user_id: str, app_config: AppConfig) -> None:
        stmt = select(AppConfigModel).where(AppConfigModel.user_id == user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            self.db_session.add(
                AppConfigModel.from_domain(user_id, app_config, self._encryption)
            )
            return
        record.update_from_domain(app_config, self._encryption)
