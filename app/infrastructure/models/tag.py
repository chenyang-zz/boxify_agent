from datetime import datetime
from typing import Self

from sqlalchemy import (
    DateTime,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.tag import Tag

from .base import Base


class TagModel(Base):
    """知识库标签 ORM 模型，标签名在单个用户内唯一。"""

    __tablename__ = "tags"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_tags_id"),
        ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_tags_user_id", ondelete="CASCADE"
        ),
        UniqueConstraint("user_id", "name", name="uq_tags_user_id_name"),
    )

    id: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
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
    def from_domain(cls, tag: Tag) -> Self:
        """从领域标签创建 ORM 记录。"""
        return cls(**tag.model_dump(mode="python"))

    def to_domain(self) -> Tag:
        """转换为领域标签模型。"""
        return Tag.model_validate(self, from_attributes=True)
