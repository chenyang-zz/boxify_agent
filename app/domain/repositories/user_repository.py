from typing import Optional, Protocol

from app.domain.models.user import User


class UserRepository(Protocol):
    """用户仓储协议"""

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """根据用户ID获取用户"""
        ...

    async def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        ...

    async def get_by_oauth_identity(
        self, provider: str, subject: str
    ) -> Optional[User]:
        """根据第三方OAuth身份获取用户"""
        ...

    async def count(self) -> int:
        """获取用户总数"""
        ...

    async def list_active_ids(self) -> list[str]:
        """获取所有启用用户ID"""
        ...

    async def save(self, user: User) -> None:
        """保存用户"""
        ...
