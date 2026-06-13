from fastapi.testclient import TestClient
import pytest

from app.application.services.auth_service import AuthService
from app.application.services.document_service import DocumentService
from app.application.services.tag_service import TagService
from app.domain.external.knowledge_search import KnowledgeSearch
from app.domain.models.document import Document
from app.domain.models.knowledge import KnowledgeChunk, KnowledgeSearchHit
from app.domain.models.tag import Tag
from app.domain.models.user import User
from app.interfaces import service_dependencies
from app.main import app


def test_notebook_routes_require_token():
    client = TestClient(app)

    response = client.get("/api/notebook/documents")

    assert response.status_code == 401
    assert response.json()["code"] == 401


def test_upload_document_creates_pending_document_for_current_user(monkeypatch):
    user_repository = InMemoryUserRepository()
    user_repository.seed_user("alice", "alice-password", user_id="user-a")
    document_repository = InMemoryDocumentRepository()
    task_dispatcher = FakeTaskDispatcher()
    storage = FakeStorage()

    def uow_factory():
        return NotebookUnitOfWork(user_repository, document_repository)

    monkeypatch.setattr(service_dependencies, "get_uow", uow_factory)
    app.dependency_overrides[service_dependencies.get_auth_service] = lambda: AuthService(
        uow_factory=uow_factory,
        secret_key="secret",
    )
    app.dependency_overrides[service_dependencies.get_document_storage] = lambda: storage
    app.dependency_overrides[service_dependencies.get_task_dispatcher] = (
        lambda: task_dispatcher
    )
    app.dependency_overrides[service_dependencies.get_knowledge_search] = (
        lambda: None
    )
    client = TestClient(app)
    token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["data"]["access_token"]

    response = client.post(
        "/api/notebook/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("notes.txt", b"hello notebook", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["file_name"] == "notes.txt"
    assert payload["status"] == "pending"
    assert payload["progress"] == 0
    assert payload["tags"] == []
    assert document_repository.saved_documents[0].user_id == "user-a"
    assert storage.saved_keys[0].startswith("notebook/user-a/")
    assert task_dispatcher.dispatched_document_ids == [payload["id"]]
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_knowledge_search_dependency_implements_knowledge_search(monkeypatch):
    async def fake_build_knowledge_search(user_id: str):
        assert user_id == "user-a"
        return FakeKnowledgeSearch()

    monkeypatch.setattr(
        service_dependencies,
        "build_knowledge_search",
        fake_build_knowledge_search,
    )
    search = await service_dependencies.get_knowledge_search(
        User(id="user-a", username="alice", password_hash="hash")
    )

    assert isinstance(search, KnowledgeSearch)


@pytest.mark.anyio
async def test_document_service_uses_knowledge_search_for_search_and_delete():
    user_repository = InMemoryUserRepository()
    document_repository = InMemoryDocumentRepository()
    tag_repository = InMemoryTagRepository()
    document = Document(user_id="user-a", file_name="notes.txt", file_key="key")
    await document_repository.save(document)
    knowledge_search = FakeKnowledgeSearch()

    def uow_factory():
        return NotebookUnitOfWork(user_repository, document_repository, tag_repository)

    tag_service = TagService(uow_factory=uow_factory, user_id="user-a")
    service = DocumentService(
        uow_factory=uow_factory,
        user_id="user-a",
        storage=FakeStorage(),
        task_dispatcher=FakeTaskDispatcher(),
        tag_service=tag_service,
        knowledge_search=knowledge_search,
    )

    hits = await service.search_documents("hello", 3, ["tag-a"])
    await service.delete_document(document.id)

    assert hits == [KnowledgeSearchHit(chunk_id="chunk-a", content="matched", score=1)]
    assert knowledge_search.search_calls == [("user-a", "hello", 3, ["tag-a"])]
    assert knowledge_search.deleted_sources == [("user-a", document.id)]


@pytest.mark.anyio
async def test_tag_service_lists_current_users_tags():
    user_repository = InMemoryUserRepository()
    document_repository = InMemoryDocumentRepository()
    tag_repository = InMemoryTagRepository(
        tags=[
            Tag(user_id="user-a", name="alpha"),
            Tag(user_id="user-b", name="beta"),
        ]
    )
    service = TagService(
        uow_factory=lambda: NotebookUnitOfWork(
            user_repository, document_repository, tag_repository
        ),
        user_id="user-a",
    )

    tags = await service.list_tags()

    assert [tag.name for tag in tags] == ["alpha"]


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


class InMemoryDocumentRepository:
    def __init__(self):
        self.saved_documents = []

    async def save(self, document):
        self.saved_documents.append(document)

    async def get_by_user(self, user_id: str, document_id: str):
        for document in self.saved_documents:
            if document.user_id == user_id and document.id == document_id:
                return document
        return None

    async def list_by_user(self, user_id: str, page: int, page_size: int, tag=None):
        docs = [doc for doc in self.saved_documents if doc.user_id == user_id]
        return docs, len(docs)

    async def delete(self, document):
        self.saved_documents.remove(document)


class NotebookUnitOfWork:
    def __init__(self, user_repository, document_repository, tag_repository=None):
        self.user = user_repository
        self.document = document_repository
        self.tag = tag_repository or InMemoryTagRepository()
        self.app_config = object()
        self.file = object()
        self.session = object()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class FakeStorage:
    def __init__(self):
        self.saved_keys = []
        self.deleted_keys = []

    async def save(self, key: str, content: bytes) -> None:
        self.saved_keys.append(key)

    async def delete(self, key: str) -> None:
        self.deleted_keys.append(key)


class FakeTaskDispatcher:
    def __init__(self):
        self.dispatched_document_ids = []

    async def dispatch_parse_document(self, document_id: str) -> None:
        self.dispatched_document_ids.append(document_id)


class FakeKnowledgeSearch:
    def __init__(self):
        self.search_calls = []
        self.deleted_sources = []
        self.saved_chunks = []

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ):
        self.search_calls.append((user_id, query, top_k, tags))
        return [KnowledgeSearchHit(chunk_id="chunk-a", content="matched", score=1)]

    async def save_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        self.saved_chunks.append(chunks)

    async def delete_by_source(self, user_id: str, document_id: str) -> None:
        self.deleted_sources.append((user_id, document_id))

    async def ensure_index(self) -> None:
        return None


class InMemoryTagRepository:
    def __init__(self, tags=None):
        self.tags = tags or []

    async def get_document_tags(self, document_id: str):
        return []

    async def list_by_user(self, user_id: str):
        return [tag for tag in self.tags if tag.user_id == user_id]
