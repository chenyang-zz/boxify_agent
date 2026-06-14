import pytest

from app.application.services.agent_service import AgentService
from app.application.services.auth_service import AuthService
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.long_term_memory import MemorySource
from app.domain.models.session import Session, SessionStatus
from app.domain.models.user import User
from app.infrastructure.repositories.db_uow import DBUnitOfWork


@pytest.mark.anyio
async def test_db_uow_clears_session_after_successful_commit():
    db_session = FakeDBSession()
    uow = DBUnitOfWork(session_factory=lambda: db_session)

    async with uow:
        assert uow.db_session is db_session
        assert uow.app_config.db_session is db_session
        assert uow.file.db_session is db_session
        assert uow.session.db_session is db_session
        assert uow.user.db_session is db_session

    assert db_session.commits == 1
    assert db_session.rollbacks == 0
    assert db_session.closes == 1
    assert uow.db_session is None


@pytest.mark.anyio
async def test_db_uow_rolls_back_and_clears_session_on_error():
    db_session = FakeDBSession()
    uow = DBUnitOfWork(session_factory=lambda: db_session)

    with pytest.raises(RuntimeError, match="boom"):
        async with uow:
            raise RuntimeError("boom")

    assert db_session.commits == 0
    assert db_session.rollbacks == 1
    assert db_session.closes == 1
    assert uow.db_session is None


@pytest.mark.anyio
async def test_db_uow_reraises_commit_failure_and_closes_session():
    db_session = FakeDBSession(commit_error=RuntimeError("commit failed"))
    uow = DBUnitOfWork(session_factory=lambda: db_session)

    with pytest.raises(RuntimeError, match="commit failed"):
        async with uow:
            pass

    assert db_session.commits == 1
    assert db_session.closes == 1
    assert uow.db_session is None


@pytest.mark.anyio
async def test_auth_service_creates_fresh_uow_for_each_operation():
    user_repository = InMemoryUserRepository()
    uow_factory = TrackingUowFactory(user_repository=user_repository)
    service = AuthService(uow_factory=uow_factory, secret_key="secret")

    await service.bootstrap_admin("admin", "password")
    await service.authenticate("admin", "password")

    assert len(uow_factory.created_uows) == 2
    assert all(uow.entered == 1 for uow in uow_factory.created_uows)
    assert all(uow.exited == 1 for uow in uow_factory.created_uows)


@pytest.mark.anyio
async def test_agent_service_persists_user_message_in_single_transaction():
    session_repository = InMemorySessionRepository()
    session = Session(id="session-1", title="对话", status=SessionStatus.PENDING)
    session_repository.sessions[session.id] = session
    uow_factory = TrackingUowFactory(session_repository=session_repository)
    task_cls = RecordingTask
    task_cls.created_tasks = []
    service = AgentService(
        uow_factory=uow_factory,
        llm=object(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=FakeSandbox,
        task_cls=task_cls,
        json_parser=object(),
        search_engine=object(),
        file_storage=object(),
    )

    async for _ in service.chat(session_id=session.id, message="hello"):
        pass

    write_uows = [
        uow for uow in uow_factory.created_uows if uow.session.latest_messages
    ]
    assert len(write_uows) == 1
    assert write_uows[0].session.added_events


@pytest.mark.anyio
async def test_agent_service_remembers_user_message_when_memory_is_configured():
    session_repository = InMemorySessionRepository()
    session = Session(id="session-1", title="对话", status=SessionStatus.PENDING)
    session_repository.sessions[session.id] = session
    uow_factory = TrackingUowFactory(session_repository=session_repository)
    memory = FakeLongTermMemoryManager()
    task_cls = RecordingTask
    task_cls.created_tasks = []
    service = AgentService(
        uow_factory=uow_factory,
        llm=object(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=FakeSandbox,
        task_cls=task_cls,
        json_parser=object(),
        search_engine=object(),
        file_storage=object(),
        memory=memory,
    )

    async for _ in service.chat(session_id=session.id, message="hello memory"):
        pass

    assert memory.remembered == [("hello memory", MemorySource.SESSION, "session-1")]


class FakeDBSession:
    def __init__(self, commit_error=None, rollback_error=None):
        self.commit_error = commit_error
        self.rollback_error = rollback_error
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    async def commit(self):
        self.commits += 1
        if self.commit_error:
            raise self.commit_error

    async def rollback(self):
        self.rollbacks += 1
        if self.rollback_error:
            raise self.rollback_error

    async def close(self):
        self.closes += 1


class TrackingUowFactory:
    def __init__(self, user_repository=None, session_repository=None):
        self.user_repository = user_repository or InMemoryUserRepository()
        self.session_repository = session_repository or InMemorySessionRepository()
        self.created_uows = []

    def __call__(self):
        uow = TrackingUnitOfWork(
            user_repository=self.user_repository,
            session_repository=self.session_repository,
        )
        self.created_uows.append(uow)
        return uow


class TrackingUnitOfWork:
    def __init__(self, user_repository, session_repository):
        self.user = user_repository
        self.session = session_repository.clone_tracker()
        self.file = object()
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited += 1
        return None


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


class InMemorySessionRepository:
    def __init__(self):
        self.sessions = {}
        self.latest_messages = []
        self.added_events = []
        self.status_updates = []

    def clone_tracker(self):
        clone = InMemorySessionRepository()
        clone.sessions = self.sessions
        return clone

    async def get_by_id(self, session_id: str):
        return self.sessions.get(session_id)

    async def save(self, session: Session):
        self.sessions[session.id] = session

    async def update_latest_message(self, session_id, message, timestamp):
        self.latest_messages.append((session_id, message, timestamp))

    async def add_event(self, session_id, event):
        self.added_events.append((session_id, event))

    async def update_unread_message_count(self, session_id, count):
        pass

    async def update_status(self, session_id, status):
        self.status_updates.append((session_id, status))


class FakeSandbox:
    id = "sandbox-1"

    @classmethod
    async def get(cls, sandbox_id):
        return None

    @classmethod
    async def create(cls):
        return cls()

    async def get_browser(self):
        return object()


class RecordingTask:
    created_tasks = []

    def __init__(self):
        self.id = "task-1"
        self.done = True
        self.input_stream = RecordingInputStream()
        self.output_stream = object()
        self.invoked = False

    @classmethod
    def get(cls, task_id):
        return None

    @classmethod
    def create(cls, task_runner):
        task = cls()
        cls.created_tasks.append(task)
        return task

    async def invoke(self):
        self.invoked = True

    def cancel(self):
        return True

    @classmethod
    async def destroy(cls):
        pass


class RecordingInputStream:
    def __init__(self):
        self.messages = []

    async def put(self, message):
        self.messages.append(message)
        return f"event-{len(self.messages)}"


class FakeLongTermMemoryManager:
    def __init__(self):
        self.remembered = []

    async def remember_text(self, content: str, source, source_session_id=None):
        self.remembered.append((content, source, source_session_id))
