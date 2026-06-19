from datetime import datetime
from typing import Any, Dict, List, Self
from uuid import uuid4
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    text,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.session import Session
from .base import Base


class SessionModel(Base):
    """会话ORM模型"""

    __tablename__ = "sessions"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="kp_sessions_id"),
        ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_sessions_user_id", ondelete="CASCADE"
        ),
        ForeignKeyConstraint(
            ["project_id"],
            ["session_projects.id"],
            name="fk_sessions_project_id",
            ondelete="CASCADE",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(255), nullable=False, primary_key=True, default=lambda: str(uuid4())
    )  # 会话id
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    project_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )
    type: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'task'::character varying")
    )
    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    sandbox_id: Mapped[str] = mapped_column(String(255), nullable=True)  # 沙箱id
    task_id: Mapped[str] = mapped_column(String(255), nullable=True)  # 任务id
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=text("''::character varying")
    )  # 会话标题
    unread_message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )  # 未读消息数
    latest_message: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''::text")
    )  # 最后一条消息
    latest_message_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=True,
    )  # 最后一条消息时间
    events: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )  # 事件列表
    files: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )  # 文件列表
    memories: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )  # 会话Agent的记忆
    status: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=text("''::character varying")
    )  # 会话状态
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )  # 更新时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP(0)")
    )  # 创建时间

    @classmethod
    def from_domain(cls, session: Session) -> Self:
        """从会话领域模型构建ORM模型"""
        return cls(
            # 基础字段: 使用BaseModal提供的python字典转换格式
            **session.model_dump(
                mode="python",
                exclude={"memories", "files", "events", "updated_at", "created_at"},
            ),
            # 复杂字段: 使用BaseModel提供的json字典转换格式
            **session.model_dump(
                mode="json",
                include={"memories", "files", "events"},
            ),
        )

    def to_domain(self) -> Session:
        """将会话ORM模型转换成领域模型"""
        return Session.model_validate(self, from_attributes=True)

    def update_from_domain(self, session: Session) -> None:
        """从传递的领域模型更新ORM数据"""
        # 基础字段: 使用BaseModal提供的python字典转换格式
        base_data = session.model_dump(
            mode="python",
            exclude={"memories", "files", "events", "updated_at", "created_at"},
        )

        # 复杂字段: 使用BaseModel提供的json字典转换格式
        json_data = session.model_dump(
            mode="json",
            include={"memories", "files", "events"},
        )

        # 合并更新
        for field, value in {**base_data, **json_data}.items():
            setattr(self, field, value)
