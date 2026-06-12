import logging
from typing import Callable, Protocol

from app.domain.models.app_config import AppConfig, create_default_app_config
from app.domain.models.user import User
from app.domain.repositories.vow import IUnitOfWork

logger = logging.getLogger(__name__)


class LegacyAppConfigRepository(Protocol):
    """旧应用配置仓储协议"""

    def exists(self) -> bool:
        """判断旧配置是否存在"""
        ...

    def load_existing(self) -> AppConfig:
        """读取已存在的旧配置"""
        ...


class AppConfigBootstrapService:
    """应用配置初始化服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        legacy_app_config_repository: LegacyAppConfigRepository,
    ) -> None:
        self._uow_factory = uow_factory
        self._legacy_app_config_repository = legacy_app_config_repository

    async def bootstrap_admin_app_config(self, admin_user: User) -> None:
        """为管理员初始化数据库应用配置"""
        async with self._uow_factory() as uow:
            existing_app_config = await uow.app_config.get_by_user_id(admin_user.id)
        if existing_app_config:
            return

        if self._legacy_app_config_repository.exists():
            try:
                app_config = self._legacy_app_config_repository.load_existing()
            except Exception as e:
                logger.warning(f"读取旧应用配置失败，管理员将使用默认应用配置: {e}")
                app_config = create_default_app_config()
        else:
            logger.warning("未找到旧应用配置文件，管理员将使用默认应用配置")
            app_config = create_default_app_config()

        async with self._uow_factory() as uow:
            existing_app_config = await uow.app_config.get_by_user_id(admin_user.id)
            if existing_app_config:
                return
            await uow.app_config.save(admin_user.id, app_config)
        logger.info("已初始化管理员应用配置")
