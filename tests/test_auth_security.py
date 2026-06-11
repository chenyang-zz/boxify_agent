from datetime import timedelta

import pytest

from app.application.errors.exceptions import UnauthorizedError
from app.application.security import PasswordHasher, TokenService
from app.application.services.auth_service import AuthService
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
