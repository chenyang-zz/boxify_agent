from datetime import timedelta

import pytest

from app.application.errors.exceptions import UnauthorizedError
from app.application.security import PasswordHasher, TokenService
from app.application.services.auth_service import AuthService
from app.application.services.oauth import OAuthStateCodec
from app.domain.models.user import User


def test_password_hasher_verifies_password_and_rejects_wrong_password():
    password_hash = PasswordHasher.hash_password("correct-password")

    assert password_hash != "correct-password"
    assert PasswordHasher.verify_password("correct-password", password_hash)
    assert not PasswordHasher.verify_password("wrong-password", password_hash)


def test_token_service_round_trips_subject_and_rejects_tampered_token():
    token_service = TokenService(secret_key="unit-test-secret")

    token = token_service.create_access_token(
        subject="user-1",
        expires_delta=timedelta(minutes=5),
    )

    assert token_service.verify_access_token(token) == "user-1"
    with pytest.raises(UnauthorizedError):
        token_service.verify_access_token(f"{token}tampered")


def test_oauth_state_codec_round_trips_and_rejects_tampered_state():
    codec = OAuthStateCodec(secret_key="unit-test-secret")

    state = codec.encode(
        provider="google",
        code_verifier="code-verifier",
        nonce="nonce-value",
    )

    decoded = codec.decode(state, expected_provider="google")
    assert decoded.provider == "google"
    assert decoded.code_verifier == "code-verifier"
    assert decoded.nonce == "nonce-value"
    tampered_state = f"{state[:12]}A{state[13:]}"
    with pytest.raises(UnauthorizedError) as exc_info:
        codec.decode(tampered_state, expected_provider="google")
    assert exc_info.value.msg == "OAuth登录状态无效或已过期"


def test_oauth_state_codec_rejects_expired_state():
    codec = OAuthStateCodec(secret_key="unit-test-secret", ttl_seconds=-1)

    state = codec.encode(
        provider="github",
        code_verifier="code-verifier",
        nonce=None,
    )

    with pytest.raises(UnauthorizedError) as exc_info:
        codec.decode(state, expected_provider="github")
    assert exc_info.value.msg == "OAuth登录状态无效或已过期"


@pytest.mark.anyio
async def test_auth_service_bootstrap_admin_creates_user_only_when_empty():
    repository = InMemoryUserRepository()
    service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository), secret_key="secret"
    )

    await service.bootstrap_admin("admin", "admin-password")
    await service.bootstrap_admin("other", "other-password")

    assert len(repository.users_by_username) == 1
    admin = repository.users_by_username["admin"]
    assert admin.is_admin
    assert admin.is_active
    assert PasswordHasher.verify_password("admin-password", admin.password_hash)


class InMemoryUserRepository:
    def __init__(self):
        self.users_by_username = {}

    async def get_by_username(self, username: str):
        return self.users_by_username.get(username)

    async def get_by_id(self, user_id: str):
        for user in self.users_by_username.values():
            if user.id == user_id:
                return user
        return None

    async def count(self) -> int:
        return len(self.users_by_username)

    async def save(self, user: User) -> None:
        self.users_by_username[user.username] = user


class InMemoryUnitOfWork:
    def __init__(self, user_repository: InMemoryUserRepository):
        self.user = user_repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
