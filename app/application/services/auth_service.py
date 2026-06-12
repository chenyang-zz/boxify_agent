from datetime import timedelta
from typing import Callable, Optional

from app.application.errors.exceptions import ForbiddenError, UnauthorizedError
from app.application.security import PasswordHasher, TokenService
from app.domain.models.user import User
from app.domain.repositories.vow import IUnitOfWork


class AuthService:
    """认证服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        secret_key: str,
        access_token_expire_minutes: int = 1440,
    ) -> None:
        self._uow_factory = uow_factory
        self._token_service = TokenService(secret_key)
        self._access_token_expire_minutes = access_token_expire_minutes

    @property
    def expires_in(self) -> int:
        """access token过期秒数"""
        return self._access_token_expire_minutes * 60

    async def bootstrap_admin(
        self, username: Optional[str], password: Optional[str]
    ) -> Optional[User]:
        """用户表为空时根据配置初始化管理员"""
        async with self._uow_factory() as uow:
            if await uow.user.count() > 0:
                return None
            if not username or not password:
                return None
            user = User(
                username=username,
                password_hash=PasswordHasher.hash_password(password),
                is_active=True,
                is_admin=True,
            )
            await uow.user.save(user)
            return user

    async def authenticate(self, username: str, password: str) -> User:
        """校验用户凭证"""
        async with self._uow_factory() as uow:
            user = await uow.user.get_by_username(username)
        if not user or not PasswordHasher.verify_password(password, user.password_hash):
            raise UnauthorizedError("用户名或密码错误")
        if not user.is_active:
            raise ForbiddenError("用户已被禁用")
        return user

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        async with self._uow_factory() as uow:
            return await uow.user.get_by_username(username)

    def create_access_token(self, user: User) -> str:
        """为用户创建access token"""
        return self._token_service.create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=self._access_token_expire_minutes),
        )

    async def get_user_by_token(self, token: str) -> User:
        """根据access token获取用户"""
        user_id = self._token_service.verify_access_token(token)
        async with self._uow_factory() as uow:
            user = await uow.user.get_by_id(user_id)
        if not user:
            raise UnauthorizedError()
        if not user.is_active:
            raise ForbiddenError("用户已被禁用")
        return user
