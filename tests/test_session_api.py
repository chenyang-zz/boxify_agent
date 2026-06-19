from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.application.services.session_service import SessionService
from app.domain.models.session import Session, SessionType
from app.domain.models.session_project import SessionProject
from app.domain.models.event import DoneEvent, MessageEvent
from app.domain.models.user import User
from app.interfaces import service_dependencies
from app.interfaces.service_dependencies import (
    get_agent_service,
    get_session_service,
    require_active_user,
    require_bearer_token,
)
from app.main import app


def test_create_session_defaults_to_independent_chat_session():
    service = FakeSessionService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    client = TestClient(app)

    response = client.post("/api/sessions", json={})

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["session_id"] == "session-1"
    assert payload["type"] == "chat"
    assert payload["project_id"] is None
    assert payload["is_pinned"] is False
    assert service.created_sessions == [("chat", None, False)]
    app.dependency_overrides.clear()


def test_create_session_accepts_pinned_flag():
    service = FakeSessionService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    client = TestClient(app)

    response = client.post("/api/sessions", json={"is_pinned": True})

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["is_pinned"] is True
    assert service.created_sessions == [("chat", None, True)]
    app.dependency_overrides.clear()


def test_sidebar_returns_projects_with_sessions_and_independent_conversations():
    service = FakeSessionService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    client = TestClient(app)

    project_response = client.post(
        "/api/sessions/projects",
        json={"name": "claude_code_src", "sort_order": 10},
    )
    sidebar_response = client.get("/api/sessions/sidebar")

    assert project_response.status_code == 200
    assert project_response.json()["data"] == {
        "project_id": "project-1",
        "name": "claude_code_src",
        "sort_order": 10,
        "is_pinned": False,
    }
    assert sidebar_response.status_code == 200
    assert sidebar_response.json()["data"] == {
        "projects": [
            {
                "project_id": "project-1",
                "name": "claude_code_src",
                "sort_order": 10,
                "is_pinned": True,
                "sessions": [
                    {
                        "session_id": "session-in-project",
                        "title": "项目会话",
                        "latest_message": "hello",
                        "latest_message_at": None,
                        "status": "pending",
                        "type": "chat",
                        "project_id": "project-1",
                        "is_pinned": True,
                    }
                ],
            }
        ],
        "standalone_conversations": [
            {
                "session_id": "session-standalone",
                "title": "独立会话",
                "latest_message": "",
                "latest_message_at": None,
                "status": "pending",
                "type": "chat",
                "project_id": None,
                "is_pinned": False,
            }
        ],
    }
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_session_service_sidebar_only_includes_chat_sessions():
    project = SessionProject(id="project-1", user_id="user-a", name="Project")
    service = SessionService(
        uow_factory=lambda: FakeSidebarUnitOfWork(
            projects=[project],
            sessions=[
                Session(
                    id="chat-in-project",
                    user_id="user-a",
                    project_id="project-1",
                    type=SessionType.CHAT,
                    title="项目聊天",
                ),
                Session(
                    id="task-in-project",
                    user_id="user-a",
                    project_id="project-1",
                    type=SessionType.TASK,
                    title="项目任务",
                ),
                Session(
                    id="chat-standalone",
                    user_id="user-a",
                    type=SessionType.CHAT,
                    title="独立聊天",
                ),
                Session(
                    id="task-standalone",
                    user_id="user-a",
                    type=SessionType.TASK,
                    title="独立任务",
                ),
            ],
        ),
        sandbox_cls=object,
        user_id="user-a",
    )

    sidebar = await service.get_sidebar()

    assert [item.project.id for item in sidebar.projects] == ["project-1"]
    assert [session.id for session in sidebar.projects[0].sessions] == [
        "chat-in-project"
    ]
    assert [session.id for session in sidebar.standalone_conversations] == [
        "chat-standalone"
    ]


def test_move_session_and_delete_project_updates_sidebar_structure():
    service = FakeSessionService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    client = TestClient(app)

    move_response = client.post(
        "/api/sessions/session-standalone/update",
        json={"project_id": "project-1"},
    )
    delete_response = client.post("/api/sessions/projects/project-1/delete")

    assert move_response.status_code == 200
    assert move_response.json()["data"]["project_id"] == "project-1"
    assert service.updated_sessions == [("session-standalone", None, "project-1", None)]
    assert delete_response.status_code == 200
    assert service.deleted_projects == ["project-1"]
    app.dependency_overrides.clear()


def test_update_session_toggles_pinned_flag():
    service = FakeSessionService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/api/sessions/session-standalone/update",
        json={"is_pinned": True},
    )

    assert response.status_code == 200
    assert response.json()["data"]["is_pinned"] is True
    assert "unread_message_count" not in response.json()["data"]
    assert service.updated_sessions == [("session-standalone", None, None, True)]
    app.dependency_overrides.clear()


def test_create_and_update_project_support_pinned_flag():
    service = FakeSessionService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    client = TestClient(app)

    create_response = client.post(
        "/api/sessions/projects",
        json={"name": "Pinned", "is_pinned": True},
    )
    update_response = client.post(
        "/api/sessions/projects/project-1/update",
        json={"is_pinned": False},
    )

    assert create_response.status_code == 200
    assert create_response.json()["data"]["is_pinned"] is True
    assert service.created_projects == [("Pinned", 0, True)]
    assert update_response.status_code == 200
    assert update_response.json()["data"]["is_pinned"] is False
    assert service.updated_projects == [("project-1", None, None, False)]
    app.dependency_overrides.clear()


def test_chat_session_uses_plain_chat_service_sse():
    service = FakeSessionService()
    chat_service = FakeChatService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    if hasattr(service_dependencies, "get_chat_service"):
        app.dependency_overrides[service_dependencies.get_chat_service] = (
            lambda: chat_service
        )
    client = TestClient(app)

    response = client.post(
        "/api/sessions/chat-session/chat",
        json={"message": "hello"},
    )

    assert response.status_code == 200
    assert "plain assistant" in response.text
    assert "agent assistant" not in response.text
    app.dependency_overrides.clear()


def test_chat_session_rejects_attachments():
    service = FakeSessionService()
    app.dependency_overrides[require_bearer_token] = lambda: object()
    app.dependency_overrides[require_active_user] = lambda: _user("user-a")
    app.dependency_overrides[get_session_service] = lambda: service
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    if hasattr(service_dependencies, "get_chat_service"):
        app.dependency_overrides[service_dependencies.get_chat_service] = (
            lambda: FakeChatService()
        )
    client = TestClient(app)

    response = client.post(
        "/api/sessions/chat-session/chat",
        json={"message": "hello", "attachments": ["file-1"]},
    )

    assert response.status_code == 400
    assert response.json()["msg"] == "普通聊天会话暂不支持附件"
    app.dependency_overrides.clear()


class FakeSessionService:
    def __init__(self):
        self.created_sessions = []
        self.created_projects = []
        self.updated_projects = []
        self.updated_sessions = []
        self.deleted_projects = []

    async def create_session(
        self, session_type="chat", project_id=None, is_pinned=False
    ):
        self.created_sessions.append((session_type, project_id, is_pinned))
        return _session(
            "session-1",
            session_type=session_type,
            project_id=project_id,
            is_pinned=is_pinned,
        )

    async def create_project(self, name, sort_order=0, is_pinned=False):
        self.created_projects.append((name, sort_order, is_pinned))
        return SimpleNamespace(
            id="project-1",
            name=name,
            sort_order=sort_order,
            is_pinned=is_pinned,
        )

    async def update_project(
        self, project_id, name=None, sort_order=None, is_pinned=None
    ):
        self.updated_projects.append((project_id, name, sort_order, is_pinned))
        return SimpleNamespace(
            id=project_id,
            name=name or "Pinned",
            sort_order=sort_order or 0,
            is_pinned=bool(is_pinned),
        )

    async def get_sidebar(self):
        return SimpleNamespace(
            projects=[
                SimpleNamespace(
                    project=SimpleNamespace(
                        id="project-1",
                        name="claude_code_src",
                        sort_order=10,
                        is_pinned=True,
                    ),
                    sessions=[
                        _session(
                            "session-in-project",
                            title="项目会话",
                            latest_message="hello",
                            project_id="project-1",
                            is_pinned=True,
                        )
                    ],
                )
            ],
            standalone_conversations=[
                _session(
                    "session-standalone",
                    title="独立会话",
                )
            ],
        )

    async def update_session(
        self, session_id, title=None, project_id=None, is_pinned=None
    ):
        self.updated_sessions.append((session_id, title, project_id, is_pinned))
        return _session(
            session_id,
            title=title or "独立会话",
            project_id=project_id,
            is_pinned=bool(is_pinned),
        )

    async def delete_project(self, project_id):
        self.deleted_projects.append(project_id)

    async def get_session(self, session_id):
        return _session(session_id, title="普通聊天", session_type="chat")


class FakeChatService:
    async def chat(self, session_id, message=None, latest_event_id=None, timestamp=None):
        yield MessageEvent(role="assistant", message="plain assistant")
        yield DoneEvent()


class FakeAgentService:
    async def chat(
        self,
        session_id,
        message=None,
        attachments=None,
        latest_event_id=None,
        timestamp=None,
    ):
        yield MessageEvent(role="assistant", message="agent assistant")
        yield DoneEvent()


class FakeSidebarUnitOfWork:
    def __init__(self, projects, sessions):
        self.session_project = FakeSidebarProjectRepository(projects)
        self.session = FakeSidebarSessionRepository(sessions)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class FakeSidebarProjectRepository:
    def __init__(self, projects):
        self.projects = projects

    async def list_by_user(self, user_id):
        return [project for project in self.projects if project.user_id == user_id]


class FakeSidebarSessionRepository:
    def __init__(self, sessions):
        self.sessions = sessions

    async def get_all_by_user(self, user_id):
        return [session for session in self.sessions if session.user_id == user_id]


def _session(
    session_id,
    title="新对话",
    latest_message="",
    session_type="chat",
    project_id=None,
    is_pinned=False,
):
    return SimpleNamespace(
        id=session_id,
        title=title,
        latest_message=latest_message,
        latest_message_at=None,
        status="pending",
        unread_message_count=0,
        type=session_type,
        project_id=project_id,
        is_pinned=is_pinned,
    )


def _user(user_id):
    return User(
        id=user_id,
        username=f"{user_id}-name",
        password_hash="unused",
        is_active=True,
        is_admin=False,
    )
