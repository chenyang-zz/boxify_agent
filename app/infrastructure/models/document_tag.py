from sqlalchemy import ForeignKeyConstraint, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DocumentTagModel(Base):
    """知识库文档标签关联 ORM 模型，依赖外键级联清理孤儿关系。"""

    __tablename__ = "document_tags"
    __table_args__ = (
        PrimaryKeyConstraint("document_id", "tag_id", name="pk_document_tags"),
        ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_tags_document_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tag_id"],
            ["tags.id"],
            name="fk_document_tags_tag_id",
            ondelete="CASCADE",
        ),
    )

    document_id: Mapped[str] = mapped_column(String(255), nullable=False)
    tag_id: Mapped[str] = mapped_column(String(255), nullable=False)
