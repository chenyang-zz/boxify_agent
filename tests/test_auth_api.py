from fastapi.testclient import TestClient

from app.application.services.auth_service import AuthService
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
