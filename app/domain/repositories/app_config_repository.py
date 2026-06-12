from typing import Optional, Protocol

from app.domain.models.app_config import AppConfig


class AppConfigRepository(Protocol):
    """应用配置仓库"""

    async def get_by_user_id(self, user_id: str) -> Optional[AppConfig]:
        """根据用户ID获取应用配置"""
        ...

    async def get_or_create_default(self, user_id: str) -> AppConfig:
        """获取用户应用配置，不存在时创建默认配置"""
        ...

    async def save(self, user_id: str, app_config: AppConfig) -> None:
        """保存用户应用配置"""
        ...
