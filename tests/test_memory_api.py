from fastapi.testclient import TestClient

from app.application.errors.exceptions import NotFoundError
from app.application.services.auth_service import AuthService
from app.application.services.memory_service import MemoryService
from app.domain.models.long_term_memory import LongTermMemory, MemoryStatus
from app.domain.models.memory_graph import (
    CommunityMember,
    CommunityRelationFact,
    CommunitySummary,
    InsightView,
    MemoryCommunityClusterStats,
    MemoryConsolidationStats,
    MemoryEntitySubgraphView,
    MemoryGraphEdgeView,
    MemoryGraphNodeView,
    MemoryGraphView,
    MemoryDuplicateMergeStats,
    MemoryProfileEntity,
    MemoryProfileGroup,
    MemoryProfileRelation,
    MemoryProfile,
    FailedMemorySnapshot,
    MemoryGraphCounts,
    MemoryQualityIssueList,
    MemoryQualityIssue,
    MemoryQualityIssueSummary,
    MemoryQualityOverview,
    MemoryReflectStats,
    MemoryRelationHistoryItem,
    MemoryTraceDialogue,
    MemoryTrace,
    MemoryTimelineEvent,
    MemoryTimelineParticipant,
)
from app.domain.models.long_term_memory import (
    MemoryReextractItem,
    MemoryReextractBatch,
)
from app.domain.models.user import User
from app.interfaces import service_dependencies
from app.main import app
from tests.test_memory_service import InMemoryMemoryRepository, MemoryUnitOfWork


def test_memory_routes_require_token():
    client = TestClient(app)

    response = client.post("/api/memories", json={"content": "我喜欢周杰伦"})
    quality_response = client.get("/api/memories/quality")
    quality_issues_response = client.get(
        "/api/memories/quality/issues?category=duplicate_entities"
    )
    detail_response = client.get("/api/memories/memory-1")
    batch_reextract_response = client.post("/api/memories/reextract")
    single_reextract_response = client.post("/api/memories/memory-1/reextract")

    assert response.status_code == 401
    assert response.json()["code"] == 401
    assert quality_response.status_code == 401
    assert quality_issues_response.status_code == 401
    assert detail_response.status_code == 401
    assert batch_reextract_response.status_code == 401
    assert single_reextract_response.status_code == 401


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


def test_list_memory_timeline_for_current_user(monkeypatch):
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
        lambda: FakeTimelineMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.get(
        "/api/memories/timeline?limit=50",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == [
        {
            "id": "event-1",
            "title": "参加周杰伦演唱会",
            "description": "用户参加了周杰伦演唱会",
            "event_time": "2026-06-15T20:00:00",
            "created_at": "2026-06-16T09:00:00",
            "participants": [
                {
                    "entity_id": "entity-1",
                    "name": "周杰伦",
                    "type": "生命体",
                }
            ],
        }
    ]
    app.dependency_overrides.clear()


def test_get_memory_graph_for_current_user(monkeypatch):
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
        lambda: FakeGraphMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.get(
        "/api/memories/graph",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "nodes": [
            {
                "id": "entity-1",
                "name": "周杰伦",
                "type": "生命体",
                "description": "歌手",
                "community_id": "community-music",
                "importance": 0.8,
                "memory_layer": "long_term",
                "access_count": 2,
                "mention_count": 3,
                "core_facts": ["用户长期喜欢周杰伦"],
                "traits": ["偏好华语流行"],
            }
        ],
        "edges": [
            {
                "source": "entity-user",
                "target": "entity-1",
                "predicate": "偏好",
                "evidence": "用户喜欢周杰伦。",
                "valid_at": None,
                "invalid_at": None,
                "is_current": True,
            }
        ],
        "communities": [
            {
                "id": "community-music",
                "name": "音乐偏好",
                "summary": "用户的音乐相关实体",
                "member_count": 1,
            }
        ],
    }
    app.dependency_overrides.clear()


def test_get_memory_entity_subgraph_for_current_user(monkeypatch):
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
        lambda: FakeGraphMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.get(
        "/api/memories/graph/entity/entity-1",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["center"] == "entity-1"
    assert response.json()["data"]["nodes"][0]["name"] == "周杰伦"
    assert response.json()["data"]["edges"][0]["predicate"] == "偏好"
    app.dependency_overrides.clear()


def test_get_memory_entity_subgraph_returns_404_for_missing_entity(monkeypatch):
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
        lambda: FakeGraphMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.get(
        "/api/memories/graph/entity/missing",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404
    assert response.json()["msg"] == "实体不存在或无权访问"
    app.dependency_overrides.clear()


def test_memory_management_endpoints_for_current_user(monkeypatch):
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
        lambda: FakeManagementMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    profile_response = client.get("/api/memories/profile", headers=headers)
    insights_response = client.get("/api/memories/insights", headers=headers)
    merge_response = client.post("/api/memories/merge-duplicates", headers=headers)
    entity_delete_response = client.post(
        "/api/memories/entities/entity-1/delete", headers=headers
    )
    insight_delete_response = client.post(
        "/api/memories/insights/insight-1/delete", headers=headers
    )
    missing_entity_response = client.post(
        "/api/memories/entities/missing/delete", headers=headers
    )
    missing_insight_response = client.post(
        "/api/memories/insights/missing/delete", headers=headers
    )
    relation_history_response = client.get(
        "/api/memories/entities/entity-1/relations/history?predicate=就职于",
        headers=headers,
    )
    missing_relation_history_response = client.get(
        "/api/memories/entities/missing/relations/history",
        headers=headers,
    )

    assert profile_response.status_code == 200
    assert profile_response.json()["data"] == {
        "total": 1,
        "type_counts": {"生命体": 1},
        "groups": [
            {
                "type": "生命体",
                "entities": [
                    {
                        "id": "entity-1",
                        "name": "周杰伦",
                        "type": "生命体",
                        "description": "歌手",
                        "community_id": "community-music",
                        "importance": 0.8,
                        "memory_layer": "long_term",
                        "access_count": 2,
                        "mention_count": 3,
                        "core_facts": ["用户长期喜欢周杰伦"],
                        "traits": ["偏好华语流行"],
                        "relations": [
                            {
                                "predicate": "偏好",
                                "target_entity_id": "entity-1",
                                "target_name": "周杰伦",
                                    "target_type": "生命体",
                                    "evidence": "用户喜欢周杰伦。",
                                    "valid_at": None,
                                    "invalid_at": None,
                                    "is_current": True,
                                }
                            ],
                    }
                ],
            }
        ],
    }
    assert insights_response.status_code == 200
    assert insights_response.json()["data"][0]["theme"] == "音乐偏好"
    assert relation_history_response.status_code == 200
    assert relation_history_response.json()["data"] == [
        {
            "relation_id": "rel-current",
            "direction": "outgoing",
            "neighbor_entity_id": "entity-company",
            "neighbor_name": "腾讯",
            "neighbor_type": "组织机构",
            "predicate": "就职于",
            "evidence": "用户现在在腾讯工作。",
            "valid_at": "2026-06-16T09:00:00",
            "invalid_at": None,
            "is_current": True,
        }
    ]
    assert merge_response.status_code == 200
    assert merge_response.json()["data"] == {
        "removed_entities": 2,
        "merged_groups": 1,
    }
    assert entity_delete_response.status_code == 200
    assert insight_delete_response.status_code == 200
    assert missing_entity_response.status_code == 404
    assert missing_entity_response.json()["msg"] == "实体不存在或无权访问"
    assert missing_insight_response.status_code == 404
    assert missing_insight_response.json()["msg"] == "洞察不存在或无权访问"
    assert missing_relation_history_response.status_code == 404
    assert missing_relation_history_response.json()["msg"] == "实体不存在或无权访问"
    app.dependency_overrides.clear()


def test_memory_quality_endpoints_for_current_user(monkeypatch):
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
        lambda: FakeQualityMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    overview_response = client.get("/api/memories/quality", headers=headers)
    issues_response = client.get(
        "/api/memories/quality/issues?category=duplicate_entities&limit=50",
        headers=headers,
    )

    assert overview_response.status_code == 200
    assert overview_response.json()["data"] == {
        "generated_at": "2026-06-18T10:00:00",
        "pg_total": 3,
        "pg_status_counts": {
            "pending": 1,
            "extracting": 0,
            "completed": 1,
            "failed": 1,
        },
        "recent_failed": [
            {
                "id": "memory-failed",
                "content": "失败记忆",
                "error_msg": "LLM timeout",
                "updated_at": "2026-06-18T09:00:00",
            }
        ],
        "graph_available": True,
        "graph_counts": {
            "dialogues": 1,
            "chunks": 2,
            "statements": 3,
            "entities": 4,
            "relations": 5,
            "events": 6,
            "involves": 7,
            "communities": 8,
            "insights": 9,
        },
        "issue_summary": {
            "duplicate_entities": 2,
            "missing_embeddings": 1,
            "orphan_entities": 0,
            "orphan_statements": 0,
            "broken_relations": 1,
            "expired_relations": 1,
            "empty_communities": 0,
            "orphan_insights": 0,
        },
    }
    assert issues_response.status_code == 200
    assert issues_response.json()["data"] == {
        "category": "duplicate_entities",
        "total": 1,
        "items": [
            {
                "category": "duplicate_entities",
                "severity": "info",
                "title": "重复实体",
                "detail": "用户/生命体 存在重复节点",
                "entity_ids": ["entity-1", "entity-dup"],
                "memory_ids": [],
                "metadata": {"name": "用户", "type": "生命体", "count": 2},
            }
        ],
    }
    app.dependency_overrides.clear()


def test_memory_detail_endpoint_for_current_user(monkeypatch):
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
        lambda: FakeDetailMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.get(
        "/api/memories/memory-1",
        headers={"Authorization": f"Bearer {token}"},
    )
    missing_response = client.get(
        "/api/memories/missing",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["id"] == "memory-1"
    assert response.json()["data"]["content"] == "用户喜欢周杰伦。"
    assert response.json()["data"]["status"] == "completed"
    assert response.json()["data"]["graph_available"] is True
    assert response.json()["data"]["trace"]["dialogue"]["id"] == "dialogue-1"
    assert missing_response.status_code == 404
    assert missing_response.json()["msg"] == "记忆不存在或无权访问"
    app.dependency_overrides.clear()


def test_memory_reextract_endpoints_for_current_user(monkeypatch):
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
        lambda: FakeReextractMemoryService()
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    batch_response = client.post(
        "/api/memories/reextract",
        headers=headers,
        json={"dry_run": True, "statuses": ["failed"], "limit": 20},
    )
    single_response = client.post(
        "/api/memories/memory-1/reextract",
        headers=headers,
    )
    missing_response = client.post(
        "/api/memories/missing/reextract",
        headers=headers,
    )

    assert batch_response.status_code == 200
    assert batch_response.json()["data"] == {
        "matched": 1,
        "dispatched": 0,
        "skipped": 1,
        "dry_run": True,
        "items": [
            {
                "memory_id": "memory-failed",
                "status": "failed",
                "graph_dialogue_id": None,
                "dispatched": False,
                "error": None,
            }
        ],
        "errors": [],
    }
    assert single_response.status_code == 200
    assert single_response.json()["data"]["dispatched"] == 1
    assert single_response.json()["data"]["items"][0]["memory_id"] == "memory-1"
    assert missing_response.status_code == 404
    assert missing_response.json()["msg"] == "记忆不存在或无权访问"
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


class FakeDetailMemoryService:
    async def detail(self, memory_id):
        from app.domain.models.long_term_memory import LongTermMemoryDetail

        if memory_id == "missing":
            raise NotFoundError("记忆不存在或无权访问")
        memory = LongTermMemory(
                id="memory-1",
                user_id="user-a",
                content="用户喜欢周杰伦。",
                status=MemoryStatus.COMPLETED,
                summary="用户喜欢周杰伦。",
        )
        return LongTermMemoryDetail(
            **memory.model_dump(mode="python"),
            graph_available=True,
            trace=MemoryTrace(
                dialogue=MemoryTraceDialogue(
                    id="dialogue-1",
                    memory_id="memory-1",
                    summary="用户喜欢周杰伦。",
                )
            ),
        )


class FakeReextractMemoryService:
    async def reextract_memory(self, memory_id):
        if memory_id == "missing":
            raise NotFoundError("记忆不存在或无权访问")
        return MemoryReextractBatch(
            matched=1,
            dispatched=1,
            skipped=0,
            dry_run=False,
            items=[
                MemoryReextractItem(
                    memory_id=memory_id,
                    status=MemoryStatus.FAILED,
                    dispatched=True,
                )
            ],
        )

    async def reextract_memories(
        self,
        memory_ids=None,
        statuses=None,
        only_missing_graph=False,
        dry_run=False,
        limit=100,
    ):
        assert memory_ids is None
        assert statuses == [MemoryStatus.FAILED]
        assert only_missing_graph is False
        assert dry_run is True
        assert limit == 20
        return MemoryReextractBatch(
            matched=1,
            dispatched=0,
            skipped=1,
            dry_run=True,
            items=[
                MemoryReextractItem(
                    memory_id="memory-failed",
                    status=MemoryStatus.FAILED,
                    dispatched=False,
                )
            ],
        )


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
            CommunitySummary(
                id="community-music",
                name="音乐偏好",
                summary="用户的音乐相关实体",
                member_count=1,
            )
        ]

    async def community_detail(self, community_id):
        return (
            [
                CommunityMember(
                    entity_id="entity-1",
                    entity_name="周杰伦",
                    entity_type="生命体",
                    description="歌手",
                    community_id=community_id,
                )
            ],
            [
                CommunityRelationFact(
                    source_entity_id="entity-user",
                    source_name="用户",
                    target_entity_id="entity-1",
                    target_name="周杰伦",
                    name="偏好",
                    evidence="用户喜欢周杰伦。",
                )
            ],
        )


class FakeTimelineMemoryService:
    async def timeline(self, limit):
        assert limit == 50
        return [
            MemoryTimelineEvent(
                id="event-1",
                title="参加周杰伦演唱会",
                description="用户参加了周杰伦演唱会",
                event_time="2026-06-15T20:00:00",
                created_at="2026-06-16T09:00:00",
                participants=[
                    MemoryTimelineParticipant(
                        entity_id="entity-1",
                        name="周杰伦",
                        type="生命体",
                    )
                ],
            )
        ]


class FakeGraphMemoryService:
    def __init__(self):
        self.nodes = [
            MemoryGraphNodeView(
                id="entity-1",
                name="周杰伦",
                type="生命体",
                description="歌手",
                community_id="community-music",
                importance=0.8,
                memory_layer="long_term",
                access_count=2,
                mention_count=3,
                core_facts=["用户长期喜欢周杰伦"],
                traits=["偏好华语流行"],
            )
        ]
        self.edges = [
            MemoryGraphEdgeView(
                source="entity-user",
                target="entity-1",
                predicate="偏好",
                evidence="用户喜欢周杰伦。",
            )
        ]
        self.communities = [
            CommunitySummary(
                id="community-music",
                name="音乐偏好",
                summary="用户的音乐相关实体",
                member_count=1,
            )
        ]

    async def graph(self):
        return MemoryGraphView(
            nodes=self.nodes,
            edges=self.edges,
            communities=self.communities,
        )

    async def entity_subgraph(self, entity_id):
        if entity_id == "missing":
            raise NotFoundError("实体不存在或无权访问")
        return MemoryEntitySubgraphView(
            center=entity_id,
            nodes=self.nodes,
            edges=self.edges,
        )


class FakeManagementMemoryService:
    async def profile(self):
        return MemoryProfile(
            total=1,
            type_counts={"生命体": 1},
            groups=[
                MemoryProfileGroup(
                    type="生命体",
                    entities=[
                        MemoryProfileEntity(
                            id="entity-1",
                            name="周杰伦",
                            type="生命体",
                            description="歌手",
                            community_id="community-music",
                            importance=0.8,
                            memory_layer="long_term",
                            access_count=2,
                            mention_count=3,
                            core_facts=["用户长期喜欢周杰伦"],
                            traits=["偏好华语流行"],
                            relations=[
                                MemoryProfileRelation(
                                    predicate="偏好",
                                    target_entity_id="entity-1",
                                    target_name="周杰伦",
                                    target_type="生命体",
                                    evidence="用户喜欢周杰伦。",
                                )
                            ],
                        )
                    ],
                )
            ],
        )

    async def list_insights(self):
        return [
            InsightView(
                id="insight-1",
                theme="音乐偏好",
                content="用户偏好华语流行音乐。",
                importance=0.8,
                confidence=0.9,
                source_count=2,
            )
        ]

    async def merge_duplicates(self):
        return MemoryDuplicateMergeStats(removed_entities=2, merged_groups=1)

    async def delete_entity(self, entity_id):
        if entity_id == "missing":
            raise NotFoundError("实体不存在或无权访问")

    async def delete_insight(self, insight_id):
        if insight_id == "missing":
            raise NotFoundError("洞察不存在或无权访问")

    async def relation_history(self, entity_id, predicate=None):
        if entity_id == "missing":
            raise NotFoundError("实体不存在或无权访问")
        assert predicate == "就职于"
        return [
            MemoryRelationHistoryItem(
                relation_id="rel-current",
                direction="outgoing",
                neighbor_entity_id="entity-company",
                neighbor_name="腾讯",
                neighbor_type="组织机构",
                predicate="就职于",
                evidence="用户现在在腾讯工作。",
                valid_at="2026-06-16T09:00:00",
                invalid_at=None,
                is_current=True,
            )
        ]


class FakeQualityMemoryService:
    async def quality(self):
        return MemoryQualityOverview(
            generated_at="2026-06-18T10:00:00",
            pg_total=3,
            pg_status_counts={
                "pending": 1,
                "extracting": 0,
                "completed": 1,
                "failed": 1,
            },
            recent_failed=[
                FailedMemorySnapshot(
                    id="memory-failed",
                    content="失败记忆",
                    error_msg="LLM timeout",
                    updated_at="2026-06-18T09:00:00",
                )
            ],
            graph_available=True,
            graph_counts=MemoryGraphCounts(
                dialogues=1,
                chunks=2,
                statements=3,
                entities=4,
                relations=5,
                events=6,
                involves=7,
                communities=8,
                insights=9,
            ),
            issue_summary=MemoryQualityIssueSummary(
                duplicate_entities=2,
                missing_embeddings=1,
                broken_relations=1,
                expired_relations=1,
            ),
        )

    async def quality_issues(self, category, limit):
        assert category == "duplicate_entities"
        assert limit == 50
        return MemoryQualityIssueList(
            category=category,
            total=1,
            items=[
                MemoryQualityIssue(
                    category=category,
                    severity="info",
                    title="重复实体",
                    detail="用户/生命体 存在重复节点",
                    entity_ids=["entity-1", "entity-dup"],
                    metadata={"name": "用户", "type": "生命体", "count": 2},
                )
            ],
        )
