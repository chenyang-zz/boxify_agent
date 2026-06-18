from datetime import timedelta
import secrets
from typing import Awaitable, Callable, Optional

from app.application.errors.exceptions import (
    BadRequestError,
    ForbiddenError,
    UnauthorizedError,
)
from app.application.security import PasswordHasher, TokenService
from app.application.services.oauth import (
    OAuthAuthorization,
    OAuthIdentity,
    OAuthProvider,
    OAuthStateCodec,
    create_code_challenge,
    create_code_verifier,
    create_nonce,
)
from app.domain.models.user import User
from app.domain.repositories.vow import IUnitOfWork


class AuthService:
    """认证服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        secret_key: str,
        access_token_expire_minutes: int = 1440,
        oauth_providers: dict[str, OAuthProvider] | None = None,
        oauth_state_codec: OAuthStateCodec | None = None,
        oauth_frontend_redirect_uri: str = "",
    ) -> None:
        self._uow_factory = uow_factory
        self._token_service = TokenService(secret_key)
        self._access_token_expire_minutes = access_token_expire_minutes
        self._oauth_providers = oauth_providers or {}
        self._oauth_state_codec = oauth_state_codec or OAuthStateCodec(secret_key)
        self._oauth_frontend_redirect_uri = oauth_frontend_redirect_uri

    @property
    def expires_in(self) -> int:
        """access token过期秒数"""
        return self._access_token_expire_minutes * 60

    @property
    def oauth_frontend_redirect_uri(self) -> str:
        """OAuth登录成功后的前端回跳地址"""
        return self._oauth_frontend_redirect_uri

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

    def create_oauth_authorization(self, provider_name: str) -> OAuthAuthorization:
        """创建第三方OAuth授权地址"""
        provider = self._get_oauth_provider(provider_name)
        code_verifier = create_code_verifier()
        nonce = create_nonce() if provider_name == "google" else None
        state = self._oauth_state_codec.encode(
            provider=provider_name,
            code_verifier=code_verifier,
            nonce=nonce,
        )
        return OAuthAuthorization(
            authorization_url=provider.build_authorization_url(
                state=state,
                code_challenge=create_code_challenge(code_verifier),
                nonce=nonce,
            )
        )

    async def authenticate_oauth_callback(
        self,
        provider_name: str,
        code: str,
        state: str,
    ) -> User:
        """处理OAuth callback并返回本地用户"""
        provider = self._get_oauth_provider(provider_name)
        oauth_state = self._oauth_state_codec.decode(
            state=state,
            expected_provider=provider_name,
        )
        identity = await provider.exchange_code_for_identity(
            code=code,
            code_verifier=oauth_state.code_verifier,
            nonce=oauth_state.nonce,
        )
        if identity.provider != provider_name:
            raise UnauthorizedError("OAuth用户信息无效")
        return await self._get_or_create_oauth_user(identity)

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

    def _get_oauth_provider(self, provider_name: str) -> OAuthProvider:
        """按名称取得 OAuth 提供方，不支持时转换为统一业务错误。"""
        provider = self._oauth_providers.get(provider_name)
        if not provider:
            raise BadRequestError("不支持的OAuth登录提供方")
        return provider

    async def _get_or_create_oauth_user(self, identity: OAuthIdentity) -> User:
        """根据第三方身份查找或创建本地用户，并同步基础资料。"""
        async with self._uow_factory() as uow:
            existing = await uow.user.get_by_oauth_identity(
                identity.provider,
                identity.subject,
            )
            if existing:
                if not existing.is_active:
                    raise ForbiddenError("用户已被禁用")
                existing.email = identity.email
                existing.avatar_url = identity.avatar_url
                await uow.user.save(existing)
                return existing

            username = await self._build_unique_oauth_username(
                identity=identity,
                user_exists=uow.user.get_by_username,
            )
            user = User(
                username=username,
                password_hash=PasswordHasher.hash_password(secrets.token_urlsafe(48)),
                is_active=True,
                is_admin=False,
                oauth_provider=identity.provider,
                oauth_subject=identity.subject,
                email=identity.email,
                avatar_url=identity.avatar_url,
            )
            await uow.user.save(user)
            return user

    async def _build_unique_oauth_username(
        self,
        identity: OAuthIdentity,
        user_exists: Callable[[str], Awaitable[object]],
    ) -> str:
        """为 OAuth 用户生成不冲突的本地用户名。"""
        base = self._normalize_username(identity.username or identity.email)
        if not base:
            base = f"{identity.provider}_{identity.subject}"
        if not await user_exists(base):
            return base

        fallback = self._normalize_username(f"{identity.provider}_{identity.subject}")
        if fallback and not await user_exists(fallback):
            return fallback

        prefix = fallback or identity.provider
        for index in range(2, 1000):
            candidate = self._normalize_username(f"{prefix}_{index}")
            if candidate and not await user_exists(candidate):
                return candidate
        raise UnauthorizedError("无法创建OAuth用户")

    @staticmethod
    def _normalize_username(value: str | None) -> str:
        """将第三方用户名清洗为本地允许的用户名片段。"""
        if not value:
            return ""
        normalized = "".join(
            char if char.isalnum() or char in {"_", "-", "."} else "_"
            for char in value.strip()
        ).strip("._-")
        return normalized[:255]
