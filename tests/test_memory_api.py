from fastapi.testclient import TestClient

from app.application.services.auth_service import AuthService
from app.application.services.memory_service import MemoryService
from app.domain.models.memory_graph import (
    CommunityMemberResult,
    CommunityRelationResult,
    CommunityResult,
    MemoryCommunityClusterStats,
    MemoryConsolidationStats,
    MemoryReflectStats,
)
from app.domain.models.user import User
from app.interfaces import service_dependencies
from app.main import app
from tests.test_memory_service import InMemoryMemoryRepository, MemoryUnitOfWork


def test_memory_routes_require_token():
    client = TestClient(app)

    response = client.post("/api/memories", json={"content": "我喜欢周杰伦"})

    assert response.status_code == 401
    assert response.json()["code"] == 401


def test_create_and_search_memory_for_current_user(monkeypatch):
    user_repository = InMemoryUserRepository()
    user_repository.seed_user("alice", "alice-password", user_id="user-a")
    memory_repository = InMemoryMemoryRepository()

    def uow_factory():
        return MemoryApiUnitOfWork(user_repository, memory_repository)

    monkeypatch.setattr(service_dependencies, "get_uow", uow_factory)
    app.dependency_overrides[service_dependencies.get_auth_service] = lambda: AuthService(
        uow_factory=uow_factory,
        secret_key="secret",
    )
    app.dependency_overrides[service_dependencies.get_memory_service] = (
        lambda: MemoryService(
            uow_factory=lambda: MemoryUnitOfWork(memory_repository),
            user_id="user-a",
        )
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    create_response = client.post(
        "/api/memories",
        headers={"Authorization": f"Bearer {token}"},
        json={"content": "我喜欢周杰伦的歌"},
    )
    search_response = client.post(
        "/api/memories/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "周杰伦", "top_k": 5},
    )

    assert create_response.status_code == 200
    assert create_response.json()["data"]["content"] == "我喜欢周杰伦的歌"
    assert create_response.json()["data"]["status"] == "pending"
    assert search_response.status_code == 200
    assert [item["content"] for item in search_response.json()["data"]] == [
        "我喜欢周杰伦的歌"
    ]
    app.dependency_overrides.clear()


def test_consolidate_memory_returns_stats_for_current_user(monkeypatch):
    user_repository = InMemoryUserRepository()
    user_repository.seed_user("alice", "alice-password", user_id="user-a")
    memory_repository = InMemoryMemoryRepository()

    def uow_factory():
        return MemoryApiUnitOfWork(user_repository, memory_repository)

    monkeypatch.setattr(service_dependencies, "get_uow", uow_factory)
    app.dependency_overrides[service_dependencies.get_auth_service] = lambda: AuthService(
        uow_factory=uow_factory,
        secret_key="secret",
    )
    app.dependency_overrides[service_dependencies.get_memory_service] = (
        lambda: FakeConsolidateMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.post(
        "/api/memories/consolidate",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "promoted_entities": 2,
        "promoted_statements": 3,
        "enhanced_profiles": 1,
    }
    app.dependency_overrides.clear()


def test_reflect_memory_returns_stats_for_current_user(monkeypatch):
    user_repository = InMemoryUserRepository()
    user_repository.seed_user("alice", "alice-password", user_id="user-a")
    memory_repository = InMemoryMemoryRepository()

    def uow_factory():
        return MemoryApiUnitOfWork(user_repository, memory_repository)

    monkeypatch.setattr(service_dependencies, "get_uow", uow_factory)
    app.dependency_overrides[service_dependencies.get_auth_service] = lambda: AuthService(
        uow_factory=uow_factory,
        secret_key="secret",
    )
    app.dependency_overrides[service_dependencies.get_memory_service] = (
        lambda: FakeReflectMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.post(
        "/api/memories/reflect",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "insights": 2,
        "skipped": None,
        "error": None,
    }
    app.dependency_overrides.clear()


def test_cluster_memory_returns_stats_for_current_user(monkeypatch):
    user_repository = InMemoryUserRepository()
    user_repository.seed_user("alice", "alice-password", user_id="user-a")
    memory_repository = InMemoryMemoryRepository()

    def uow_factory():
        return MemoryApiUnitOfWork(user_repository, memory_repository)

    monkeypatch.setattr(service_dependencies, "get_uow", uow_factory)
    app.dependency_overrides[service_dependencies.get_auth_service] = lambda: AuthService(
        uow_factory=uow_factory,
        secret_key="secret",
    )
    app.dependency_overrides[service_dependencies.get_memory_service] = (
        lambda: FakeCommunityMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.post(
        "/api/memories/cluster",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "communities": 2,
        "assigned_entities": 3,
        "merged_communities": 1,
        "enhanced_communities": 2,
        "skipped": None,
        "error": None,
    }
    app.dependency_overrides.clear()


def test_list_and_detail_memory_communities_for_current_user(monkeypatch):
    user_repository = InMemoryUserRepository()
    user_repository.seed_user("alice", "alice-password", user_id="user-a")
    memory_repository = InMemoryMemoryRepository()

    def uow_factory():
        return MemoryApiUnitOfWork(user_repository, memory_repository)

    monkeypatch.setattr(service_dependencies, "get_uow", uow_factory)
    app.dependency_overrides[service_dependencies.get_auth_service] = lambda: AuthService(
        uow_factory=uow_factory,
        secret_key="secret",
    )
    app.dependency_overrides[service_dependencies.get_memory_service] = (
        lambda: FakeCommunityMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    list_response = client.get(
        "/api/memories/communities",
        headers={"Authorization": f"Bearer {token}"},
    )
    detail_response = client.get(
        "/api/memories/communities/community-music",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert list_response.status_code == 200
    assert list_response.json()["data"] == [
        {
            "id": "community-music",
            "name": "音乐偏好",
            "summary": "用户的音乐相关实体",
            "member_count": 1,
        }
    ]
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["members"][0]["entity_name"] == "周杰伦"
    assert detail_response.json()["data"]["relationships"][0]["name"] == "偏好"
    app.dependency_overrides.clear()


class InMemoryUserRepository:
    def __init__(self):
        self.users_by_username = {}

    def seed_user(self, username: str, password: str, user_id: str):
        from app.application.security import PasswordHasher

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


class MemoryApiUnitOfWork:
    def __init__(self, user_repository, memory_repository):
        self.user = user_repository
        self.memory = memory_repository
        self.app_config = object()
        self.document = object()
        self.file = object()
        self.session = object()
        self.tag = object()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class FakeConsolidateMemoryService:
    async def consolidate(self):
        return MemoryConsolidationStats(
            promoted_entities=2,
            promoted_statements=3,
            enhanced_profiles=1,
        )


class FakeReflectMemoryService:
    async def reflect(self):
        return MemoryReflectStats(insights=2)


class FakeCommunityMemoryService:
    async def cluster(self):
        return MemoryCommunityClusterStats(
            communities=2,
            assigned_entities=3,
            merged_communities=1,
            enhanced_communities=2,
        )

    async def list_communities(self):
        return [
            CommunityResult(
                id="community-music",
                name="音乐偏好",
                summary="用户的音乐相关实体",
                member_count=1,
            )
        ]

    async def community_detail(self, community_id):
        return (
            [
                CommunityMemberResult(
                    entity_id="entity-1",
                    entity_name="周杰伦",
                    entity_type="生命体",
                    description="歌手",
                    community_id=community_id,
                )
            ],
            [
                CommunityRelationResult(
                    source_entity_id="entity-user",
                    source_name="用户",
                    target_entity_id="entity-1",
                    target_name="周杰伦",
                    name="偏好",
                    evidence="用户喜欢周杰伦。",
                )
            ],
        )
