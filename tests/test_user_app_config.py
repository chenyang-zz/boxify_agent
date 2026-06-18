import pytest
from fastapi.testclient import TestClient

from app.application.security import PasswordHasher
from app.application.services.auth_service import AuthService
from app.application.services.app_config_service import AppConfigService
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
    NotebookConfig,
    NotebookEmbeddingConfig,
)
from app.domain.models.user import User
from app.main import app
from app.interfaces import service_dependencies


@pytest.mark.anyio
async def test_app_config_service_isolates_config_by_user():
    app_config_repository = InMemoryAppConfigRepository()
    uow_factory = TrackingUowFactory(app_config_repository=app_config_repository)
    user_a_service = AppConfigService(uow_factory=uow_factory, user_id="user-a")
    user_b_service = AppConfigService(uow_factory=uow_factory, user_id="user-b")

    await user_a_service.update_llm_config(
        LLMConfig(
            base_url="https://user-a.example.com",
            api_key="secret-a",
            model_name="user-a-model",
        )
    )

    user_a_llm_config = await user_a_service.get_llm_config()
    user_b_llm_config = await user_b_service.get_llm_config()

    assert user_a_llm_config.base_url == "https://user-a.example.com"
    assert user_a_llm_config.model_name == "user-a-model"
    assert user_b_llm_config.base_url == LLMConfig().base_url
    assert user_b_llm_config.model_name == LLMConfig().model_name


@pytest.mark.anyio
async def test_update_llm_empty_api_key_preserves_only_current_users_key():
    app_config_repository = InMemoryAppConfigRepository()
    await app_config_repository.save(
        "user-a",
        AppConfig(
            llm_config=LLMConfig(api_key="secret-a", model_name="old-a"),
            agent_config=AgentConfig(),
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
        ),
    )
    await app_config_repository.save(
        "user-b",
        AppConfig(
            llm_config=LLMConfig(api_key="secret-b", model_name="old-b"),
            agent_config=AgentConfig(),
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
        ),
    )
    service = AppConfigService(
        uow_factory=TrackingUowFactory(app_config_repository=app_config_repository),
        user_id="user-a",
    )

    await service.update_llm_config(LLMConfig(api_key="", model_name="new-a"))

    user_a_config = await app_config_repository.get_by_user_id("user-a")
    user_b_config = await app_config_repository.get_by_user_id("user-b")
    assert user_a_config.llm_config.api_key == "secret-a"
    assert user_a_config.llm_config.model_name == "new-a"
    assert user_b_config.llm_config.api_key == "secret-b"
    assert user_b_config.llm_config.model_name == "old-b"


@pytest.mark.anyio
async def test_update_notebook_embedding_empty_api_key_preserves_current_users_key():
    app_config_repository = InMemoryAppConfigRepository()
    await app_config_repository.save(
        "user-a",
        AppConfig(
            llm_config=LLMConfig(),
            agent_config=AgentConfig(),
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
            notebook_config=NotebookConfig(
                embedding_config=NotebookEmbeddingConfig(
                    api_key="embedding-secret-a",
                    model_name="old-embedding-a",
                )
            ),
        ),
    )
    service = AppConfigService(
        uow_factory=TrackingUowFactory(app_config_repository=app_config_repository),
        user_id="user-a",
    )

    updated = await service.update_notebook_embedding_config(
        NotebookEmbeddingConfig(api_key="", model_name="new-embedding-a")
    )

    saved = await app_config_repository.get_by_user_id("user-a")
    assert updated.api_key == "embedding-secret-a"
    assert updated.model_name == "new-embedding-a"
    assert saved.notebook_config.embedding_config.api_key == "embedding-secret-a"


@pytest.mark.anyio
async def test_app_config_service_load_and_save_share_one_uow():
    app_config_repository = InMemoryAppConfigRepository()
    uow_factory = TrackingUowFactory(app_config_repository=app_config_repository)
    service = AppConfigService(uow_factory=uow_factory, user_id="user-a")

    await service.update_agent_config(AgentConfig(max_iterations=42))

    write_uows = [uow for uow in uow_factory.created_uows if uow.app_config.saves]
    assert len(write_uows) == 1
    assert write_uows[0].app_config.loads == ["user-a"]
    assert write_uows[0].app_config.saves == ["user-a"]


def test_agent_config_defaults_active_recall_enabled():
    assert AgentConfig().enable_active_recall is True
    assert AgentConfig.model_validate({}).enable_active_recall is True


@pytest.mark.anyio
async def test_get_agent_service_uses_current_users_database_config(monkeypatch):
    app_config_repository = InMemoryAppConfigRepository()
    await app_config_repository.save(
        "user-a",
        AppConfig(
            llm_config=LLMConfig(
                base_url="https://user-a.example.com",
                api_key="secret-a",
                model_name="user-a-model",
            ),
            agent_config=AgentConfig(max_iterations=42),
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
        ),
    )
    captured_llm_configs = []

    class FakeLLM:
        def __init__(self, llm_config):
            captured_llm_configs.append(llm_config)

    monkeypatch.setattr(service_dependencies, "OpenAILLM", FakeLLM)
    monkeypatch.setattr(service_dependencies, "CosFileStorage", lambda **kwargs: object())
    monkeypatch.setattr(
        service_dependencies,
        "get_uow",
        lambda: TrackingUnitOfWork(app_config_repository=app_config_repository),
    )

    user = User(
        id="user-a",
        username="alice",
        password_hash="hash",
        is_active=True,
        is_admin=False,
    )

    agent_service = await service_dependencies.get_agent_service(
        cos=object(), current_user=user
    )

    assert agent_service._agent_config.max_iterations == 42
    assert captured_llm_configs[0].base_url == "https://user-a.example.com"
    assert captured_llm_configs[0].model_name == "user-a-model"


@pytest.mark.anyio
async def test_get_agent_service_skips_active_recall_when_disabled(monkeypatch):
    app_config_repository = InMemoryAppConfigRepository()
    await app_config_repository.save(
        "user-a",
        AppConfig(
            llm_config=LLMConfig(),
            agent_config=AgentConfig(enable_active_recall=False),
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
        ),
    )

    async def fake_memory_graph(user_id):
        assert user_id == "user-a"
        return object(), object()

    class FakeActiveRecall:
        def __init__(self, **kwargs):
            raise AssertionError("active recall should not be built")

    monkeypatch.setattr(service_dependencies, "OpenAILLM", lambda config: object())
    monkeypatch.setattr(service_dependencies, "CosFileStorage", lambda **kwargs: object())
    monkeypatch.setattr(service_dependencies, "_build_optional_memory_graph", fake_memory_graph)
    monkeypatch.setattr(service_dependencies, "MemoryActiveRecall", FakeActiveRecall)
    monkeypatch.setattr(
        service_dependencies,
        "get_uow",
        lambda: TrackingUnitOfWork(app_config_repository=app_config_repository),
    )

    agent_service = await service_dependencies.get_agent_service(
        cos=object(),
        current_user=User(
            id="user-a",
            username="alice",
            password_hash="hash",
            is_active=True,
            is_admin=False,
        ),
    )

    assert agent_service._active_recall is None


def test_app_config_routes_isolate_config_by_logged_in_user(monkeypatch):
    user_repository = InMemoryUserRepository()
    user_repository.seed_user("alice", "alice-password", user_id="user-a")
    user_repository.seed_user("bob", "bob-password", user_id="user-b")
    app_config_repository = InMemoryAppConfigRepository()

    def uow_factory():
        return AuthConfigUnitOfWork(
            user_repository=user_repository,
            app_config_repository=app_config_repository,
        )

    monkeypatch.setattr(service_dependencies, "get_uow", uow_factory)
    app.dependency_overrides[service_dependencies.get_auth_service] = lambda: AuthService(
        uow_factory=uow_factory,
        secret_key="secret",
    )
    client = TestClient(app)

    alice_token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]
    bob_token = client.post(
        "/api/auth/login",
        json={"username": "bob", "password": "bob-password"},
    ).json()["data"]["access_token"]

    alice_update_response = client.post(
        "/api/app-config/llm",
        headers={"Authorization": f"Bearer {alice_token}"},
        json={
            "base_url": "https://alice.example.com",
            "api_key": "alice-secret",
            "model_name": "alice-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    )
    bob_get_response = client.get(
        "/api/app-config/llm",
        headers={"Authorization": f"Bearer {bob_token}"},
    )

    assert alice_update_response.status_code == 200
    assert "alice-secret" not in alice_update_response.text
    assert bob_get_response.status_code == 200
    assert bob_get_response.json()["data"]["base_url"] == LLMConfig().base_url
    assert bob_get_response.json()["data"]["model_name"] == LLMConfig().model_name
    app.dependency_overrides.clear()


class InMemoryAppConfigRepository:
    def __init__(self):
        self.configs_by_user_id = {}

    async def get_by_user_id(self, user_id: str):
        return self.configs_by_user_id.get(user_id)

    async def get_or_create_default(self, user_id: str):
        app_config = self.configs_by_user_id.get(user_id)
        if not app_config:
            app_config = AppConfig(
                llm_config=LLMConfig(),
                agent_config=AgentConfig(),
                mcp_config=MCPConfig(),
                a2a_config=A2AConfig(),
                notebook_config=NotebookConfig(),
            )
            self.configs_by_user_id[user_id] = app_config
        return app_config

    async def save(self, user_id: str, app_config: AppConfig) -> None:
        self.configs_by_user_id[user_id] = app_config


class InMemoryUserRepository:
    def __init__(self):
        self.users_by_username = {}

    def seed_user(self, username: str, password: str, user_id: str):
        user = User(
            id=user_id,
            username=username,
            password_hash=PasswordHasher.hash_password(password),
            is_active=True,
            is_admin=False,
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


class AuthConfigUnitOfWork:
    def __init__(
        self,
        user_repository: InMemoryUserRepository,
        app_config_repository: InMemoryAppConfigRepository,
    ):
        self.user = user_repository
        self.app_config = app_config_repository
        self.session = object()
        self.file = object()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TrackingAppConfigRepository:
    def __init__(self, repository: InMemoryAppConfigRepository):
        self.repository = repository
        self.loads = []
        self.saves = []

    async def get_by_user_id(self, user_id: str):
        self.loads.append(user_id)
        return await self.repository.get_by_user_id(user_id)

    async def get_or_create_default(self, user_id: str):
        self.loads.append(user_id)
        return await self.repository.get_or_create_default(user_id)

    async def save(self, user_id: str, app_config: AppConfig) -> None:
        self.saves.append(user_id)
        await self.repository.save(user_id, app_config)


class TrackingUowFactory:
    def __init__(self, app_config_repository: InMemoryAppConfigRepository):
        self.app_config_repository = app_config_repository
        self.created_uows = []

    def __call__(self):
        uow = TrackingUnitOfWork(app_config_repository=self.app_config_repository)
        self.created_uows.append(uow)
        return uow


class TrackingUnitOfWork:
    def __init__(self, app_config_repository: InMemoryAppConfigRepository):
        self.app_config = TrackingAppConfigRepository(app_config_repository)
        self.user = object()
        self.session = object()
        self.file = object()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
