from datetime import datetime
from typing import Self

from sqlalchemy import DateTime, ForeignKeyConstraint, PrimaryKeyConstraint, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.long_term_memory import LongTermMemory

from .base import Base


class MemoryModel(Base):
    """长期记忆 ORM 模型。"""

    __tablename__ = "memories"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_memories_id"),
        ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_memories_user_id", ondelete="CASCADE"
        ),
    )

    id: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    source_session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    graph_dialogue_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    graph_stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    def from_domain(cls, memory: LongTermMemory) -> Self:
        """从领域模型创建 ORM 记录。"""
        return cls(**memory.model_dump(mode="python", exclude={"graph_data"}))

    def to_domain(self) -> LongTermMemory:
        """转换为领域模型。"""
        return LongTermMemory.model_validate(self, from_attributes=True)

    def update_from_domain(self, memory: LongTermMemory) -> None:
        """用领域模型覆盖 ORM 字段。"""
        for field, value in memory.model_dump(
            mode="python", exclude={"graph_data"}
        ).items():
            setattr(self, field, value)
