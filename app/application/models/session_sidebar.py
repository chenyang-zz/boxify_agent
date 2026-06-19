from dataclasses import dataclass

from app.domain.models.session import Session
from app.domain.models.session_project import SessionProject


@dataclass
class SidebarProjectView:
    """侧边栏项目及其会话。"""

    project: SessionProject
    sessions: list[Session]


@dataclass
class SessionSidebarView:
    """侧边栏组合查询结果。"""

    projects: list[SidebarProjectView]
    standalone_conversations: list[Session]
