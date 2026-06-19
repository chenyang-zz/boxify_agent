from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from app.application.services.auth_service import AuthService
from app.application.services.oauth import OAuthIdentity
from app.domain.models.health_status import HealthStatus
from app.domain.models.user import User
from app.interfaces.service_dependencies import get_auth_service, get_status_service
from app.main import app


def test_login_returns_bearer_token_and_current_user():
    repository = InMemoryUserRepository()
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository), secret_key="secret"
    )
    repository.seed_user("admin", "admin-password")
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)

    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin-password"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["data"]["token_type"] == "bearer"
    assert payload["data"]["access_token"]
    assert payload["data"]["expires_in"] > 0
    assert payload["data"]["user"]["username"] == "admin"
    assert "password" not in str(payload["data"]).lower()

    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {payload['data']['access_token']}"},
    )

    assert me_response.status_code == 200
    assert me_response.json()["data"]["username"] == "admin"
    app.dependency_overrides.clear()


def test_me_returns_profile_fields_for_current_user():
    repository = InMemoryUserRepository()
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository), secret_key="secret"
    )
    user = repository.seed_user("admin", "admin-password")
    user.email = "admin@example.com"
    user.avatar_url = "https://example.com/admin.png"
    user.oauth_provider = "google"
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)
    token = auth_service.create_access_token(user)

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == user.id
    assert data["username"] == "admin"
    assert data["email"] == "admin@example.com"
    assert data["avatar_url"] == "https://example.com/admin.png"
    assert data["oauth_provider"] == "google"
    assert data["created_at"]
    assert data["updated_at"]
    assert "password" not in str(data).lower()
    assert "oauth_subject" not in data
    app.dependency_overrides.clear()


def test_login_rejects_invalid_credentials_with_uniform_401_response():
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(InMemoryUserRepository()),
        secret_key="secret",
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)

    response = client.post(
        "/api/auth/login",
        json={"username": "missing", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json()["code"] == 401
    assert response.json()["msg"] == "用户名或密码错误"
    app.dependency_overrides.clear()


def test_validation_error_returns_sanitized_field_errors():
    app.dependency_overrides[get_auth_service] = lambda: AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(InMemoryUserRepository()),
        secret_key="secret",
    )
    client = TestClient(app)

    response = client.post(
        "/api/auth/login",
        json={"password": "secret-password-should-not-echo"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == 422
    assert payload["msg"] == "请求参数数据校验错误，请核实后重试"
    assert payload["data"]["errors"] == [
        {
            "field": "body.username",
            "message": "Field required",
            "type": "missing",
        }
    ]
    assert "secret-password-should-not-echo" not in response.text
    app.dependency_overrides.clear()


def test_me_requires_token():
    client = TestClient(app)

    response = client.get("/api/auth/me")

    assert response.status_code == 401
    assert response.json()["code"] == 401


def test_oauth_authorize_returns_github_authorization_url_with_state_and_pkce():
    repository = InMemoryUserRepository()
    github = FakeOAuthProvider(
        provider="github",
        authorization_endpoint="https://github.test/login/oauth/authorize",
        scope="read:user user:email",
    )
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository),
        secret_key="secret",
        oauth_providers={"github": github},
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)

    response = client.get("/api/auth/oauth/github/authorize")

    assert response.status_code == 200
    payload = response.json()
    authorization_url = payload["data"]["authorization_url"]
    parsed = urlparse(authorization_url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "github.test"
    assert query["scope"] == ["read:user user:email"]
    assert query["state"][0]
    assert query["code_challenge"][0]
    assert query["code_challenge_method"] == ["S256"]
    app.dependency_overrides.clear()


def test_oauth_callback_creates_google_user_and_returns_bearer_token():
    repository = InMemoryUserRepository()
    google = FakeOAuthProvider(
        provider="google",
        authorization_endpoint="https://accounts.google.test/o/oauth2/v2/auth",
        scope="openid email profile",
        identity=OAuthIdentity(
            provider="google",
            subject="google-sub-123",
            username="alice",
            email="alice@example.com",
            avatar_url="https://example.com/alice.png",
        ),
    )
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository),
        secret_key="secret",
        oauth_providers={"google": google},
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)
    state = parse_qs(
        urlparse(
            client.get("/api/auth/oauth/google/authorize")
            .json()["data"]["authorization_url"]
        ).query
    )["state"][0]

    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "google-code", "state": state},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["token_type"] == "bearer"
    assert payload["data"]["access_token"]
    assert payload["data"]["user"]["username"] == "alice"
    user = repository.users_by_username["alice"]
    assert user.oauth_provider == "google"
    assert user.oauth_subject == "google-sub-123"
    assert user.email == "alice@example.com"
    assert google.received_code == "google-code"
    app.dependency_overrides.clear()


def test_oauth_callback_redirects_to_frontend_when_redirect_uri_is_configured():
    repository = InMemoryUserRepository()
    github = FakeOAuthProvider(
        provider="github",
        authorization_endpoint="https://github.test/login/oauth/authorize",
        scope="read:user user:email",
        identity=OAuthIdentity(
            provider="github",
            subject="github-123",
            username="octo",
            email="octo@example.com",
            avatar_url=None,
        ),
    )
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository),
        secret_key="secret",
        oauth_providers={"github": github},
        oauth_frontend_redirect_uri="https://app.boxify.test/oauth/callback",
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)
    state = parse_qs(
        urlparse(
            client.get("/api/auth/oauth/github/authorize")
            .json()["data"]["authorization_url"]
        ).query
    )["state"][0]

    response = client.get(
        "/api/auth/oauth/github/callback",
        params={"code": "github-code", "state": state},
        follow_redirects=False,
    )

    assert response.status_code == 302
    location = response.headers["location"]
    parsed = urlparse(location)
    fragment = parse_qs(parsed.fragment)
    assert parsed.scheme == "https"
    assert parsed.netloc == "app.boxify.test"
    assert parsed.path == "/oauth/callback"
    assert fragment["token_type"] == ["bearer"]
    assert fragment["access_token"][0]
    assert fragment["expires_in"] == ["86400"]
    assert fragment["user_id"] == [repository.users_by_username["octo"].id]
    assert fragment["username"] == ["octo"]
    app.dependency_overrides.clear()


def test_oauth_callback_reuses_existing_identity_without_creating_duplicate_user():
    repository = InMemoryUserRepository()
    existing = repository.seed_oauth_user(
        username="octo",
        provider="github",
        subject="42",
        email="octo@example.com",
    )
    github = FakeOAuthProvider(
        provider="github",
        authorization_endpoint="https://github.test/login/oauth/authorize",
        scope="read:user user:email",
        identity=OAuthIdentity(
            provider="github",
            subject="42",
            username="octo-new",
            email="octo-new@example.com",
            avatar_url=None,
        ),
    )
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository),
        secret_key="secret",
        oauth_providers={"github": github},
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)
    state = parse_qs(
        urlparse(
            client.get("/api/auth/oauth/github/authorize")
            .json()["data"]["authorization_url"]
        ).query
    )["state"][0]

    response = client.get(
        "/api/auth/oauth/github/callback",
        params={"code": "github-code", "state": state},
    )

    assert response.status_code == 200
    assert response.json()["data"]["user"]["id"] == existing.id
    assert len(repository.users_by_username) == 1
    app.dependency_overrides.clear()


def test_oauth_callback_rejects_inactive_existing_user():
    repository = InMemoryUserRepository()
    repository.seed_oauth_user(
        username="disabled",
        provider="google",
        subject="disabled-sub",
        email="disabled@example.com",
        is_active=False,
    )
    google = FakeOAuthProvider(
        provider="google",
        authorization_endpoint="https://accounts.google.test/o/oauth2/v2/auth",
        scope="openid email profile",
        identity=OAuthIdentity(
            provider="google",
            subject="disabled-sub",
            username="disabled",
            email="disabled@example.com",
            avatar_url=None,
        ),
    )
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(repository),
        secret_key="secret",
        oauth_providers={"google": google},
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)
    state = parse_qs(
        urlparse(
            client.get("/api/auth/oauth/google/authorize")
            .json()["data"]["authorization_url"]
        ).query
    )["state"][0]

    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "google-code", "state": state},
    )

    assert response.status_code == 403
    assert response.json()["msg"] == "用户已被禁用"
    app.dependency_overrides.clear()


def test_oauth_callback_rejects_tampered_state():
    auth_service = AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(InMemoryUserRepository()),
        secret_key="secret",
        oauth_providers={
            "github": FakeOAuthProvider(
                provider="github",
                authorization_endpoint="https://github.test/login/oauth/authorize",
                scope="read:user user:email",
            )
        },
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)

    response = client.get(
        "/api/auth/oauth/github/callback",
        params={"code": "github-code", "state": "not-a-valid-state"},
    )

    assert response.status_code == 401
    assert response.json()["msg"] == "OAuth登录状态无效或已过期"
    app.dependency_overrides.clear()


def test_oauth_unknown_provider_returns_400():
    app.dependency_overrides[get_auth_service] = lambda: AuthService(
        uow_factory=lambda: InMemoryUnitOfWork(InMemoryUserRepository()),
        secret_key="secret",
    )
    client = TestClient(app)

    response = client.get("/api/auth/oauth/twitter/authorize")

    assert response.status_code == 400
    assert response.json()["msg"] == "不支持的OAuth登录提供方"
    app.dependency_overrides.clear()


def test_status_is_public_and_business_routes_require_token():
    app.dependency_overrides[get_status_service] = lambda: FakeStatusService()
    client = TestClient(app)

    status_response = client.get("/api/status")
    config_response = client.get("/api/app-config/agent")
    sessions_response = client.get("/api/sessions")

    assert status_response.status_code != 401
    assert config_response.status_code == 401
    assert sessions_response.status_code == 401
    app.dependency_overrides.clear()


def test_status_returns_500_when_elasticsearch_checker_fails():
    app.dependency_overrides[get_status_service] = lambda: FakeStatusService(
        [
            HealthStatus(service="postgres", status="ok"),
            HealthStatus(service="redis", status="ok"),
            HealthStatus(
                service="elasticsearch",
                status="error",
                details="Elasticsearch服务Ping失败",
            ),
        ]
    )
    client = TestClient(app)

    response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 500
    assert payload["msg"] == "系统存在服务异常"
    assert payload["data"][2]["service"] == "elasticsearch"
    assert payload["data"][2]["status"] == "error"
    app.dependency_overrides.clear()


class FakeStatusService:
    def __init__(self, statuses=None):
        self._statuses = statuses or []

    async def check_all(self):
        return self._statuses


class FakeOAuthProvider:
    def __init__(
        self,
        provider: str,
        authorization_endpoint: str,
        scope: str,
        identity: OAuthIdentity | None = None,
    ):
        self.provider = provider
        self.authorization_endpoint = authorization_endpoint
        self.scope = scope
        self.identity = identity or OAuthIdentity(
            provider=provider,
            subject=f"{provider}-subject",
            username=f"{provider}-user",
            email=f"{provider}@example.com",
            avatar_url=None,
        )
        self.received_code = None

    def build_authorization_url(
        self,
        state: str,
        code_challenge: str,
        nonce: str | None,
    ) -> str:
        from urllib.parse import urlencode

        params = {
            "response_type": "code",
            "scope": self.scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if nonce:
            params["nonce"] = nonce
        return f"{self.authorization_endpoint}?{urlencode(params)}"

    async def exchange_code_for_identity(
        self,
        code: str,
        code_verifier: str,
        nonce: str | None,
    ) -> OAuthIdentity:
        self.received_code = code
        assert code_verifier
        return self.identity


class InMemoryUserRepository:
    def __init__(self):
        self.users_by_username = {}

    def seed_user(self, username: str, password: str):
        from app.application.security import PasswordHasher

        user = User(
            username=username,
            password_hash=PasswordHasher.hash_password(password),
            is_active=True,
            is_admin=True,
        )
        self.users_by_username[username] = user
        return user

    def seed_oauth_user(
        self,
        username: str,
        provider: str,
        subject: str,
        email: str | None,
        is_active: bool = True,
    ):
        from app.application.security import PasswordHasher

        user = User(
            username=username,
            password_hash=PasswordHasher.hash_password("unused-password"),
            is_active=is_active,
            is_admin=False,
            oauth_provider=provider,
            oauth_subject=subject,
            email=email,
        )
        self.users_by_username[username] = user
        return user

    async def get_by_username(self, username: str):
        return self.users_by_username.get(username)

    async def get_by_oauth_identity(self, provider: str, subject: str):
        for user in self.users_by_username.values():
            if user.oauth_provider == provider and user.oauth_subject == subject:
                return user
        return None

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
