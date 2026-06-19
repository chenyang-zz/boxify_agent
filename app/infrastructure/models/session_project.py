from datetime import datetime
from typing import Self

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.session_project import SessionProject

from .base import Base


class SessionProjectModel(Base):
    """会话项目 ORM 模型。"""

    __tablename__ = "session_projects"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_session_projects_id"),
        ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_session_projects_user_id",
            ondelete="CASCADE",
        ),
        UniqueConstraint("user_id", "name", name="uq_session_projects_user_id_name"),
    )

    id: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sort_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP(0)")
    )

    @classmethod
    def from_domain(cls, project: SessionProject) -> Self:
        """从领域项目创建 ORM 记录。"""
        return cls(**project.model_dump(mode="python"))

    def to_domain(self) -> SessionProject:
        """转换为领域项目模型。"""
        return SessionProject.model_validate(self, from_attributes=True)

    def update_from_domain(self, project: SessionProject) -> None:
        """从领域项目更新 ORM 记录。"""
        for field, value in project.model_dump(
            mode="python",
            exclude={"created_at", "updated_at"},
        ).items():
            setattr(self, field, value)
