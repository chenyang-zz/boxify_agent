from datetime import datetime
from typing import Self

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.document import Document

from .base import Base


class DocumentModel(Base):
    """知识库文档 ORM 模型，表名不带 notebook 前缀以保持业务通用命名。"""

    __tablename__ = "documents"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_documents_id"),
        ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_documents_user_id", ondelete="CASCADE"
        ),
    )

    id: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_ext: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="file")
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True, default="pending"
    )
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    chunk_num: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
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
    def from_domain(cls, document: Document) -> Self:
        """从领域模型创建 ORM 记录。"""
        return cls(**document.model_dump(mode="python"))

    def to_domain(self) -> Document:
        """转换为领域模型，供应用层避免感知 ORM 类型。"""
        return Document.model_validate(self, from_attributes=True)

    def update_from_domain(self, document: Document) -> None:
        """用领域模型覆盖 ORM 字段，保持 save 语义为 upsert。"""
        for field, value in document.model_dump(mode="python").items():
            setattr(self, field, value)
