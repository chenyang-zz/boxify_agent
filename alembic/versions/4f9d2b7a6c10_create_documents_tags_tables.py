"""create documents tags tables

Revision ID: 4f9d2b7a6c10
Revises: bc7e4c1a2d03
Create Date: 2026-06-12 21:15:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "4f9d2b7a6c10"
down_revision: Union[str, None] = "bc7e4c1a2d03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 为用户独立 AppConfig 增加 notebook 配置，不从 YAML 运行时读取。
    op.add_column(
        "app_configs",
        sa.Column(
            "notebook_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    # documents/tags/document_tags 通过 users.id 外键实现用户级知识库隔离。
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("file_name", sa.String(length=512), nullable=False),
        sa.Column("file_ext", sa.String(length=32), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("file_key", sa.String(length=1024), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("chunk_num", sa.Integer(), nullable=False),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_documents_user_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents_id"),
    )
    op.create_index(op.f("ix_documents_status"), "documents", ["status"], unique=False)
    op.create_index(op.f("ix_documents_user_id"), "documents", ["user_id"], unique=False)
    op.create_table(
        "tags",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_tags_user_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tags_id"),
        sa.UniqueConstraint("user_id", "name", name="uq_tags_user_id_name"),
    )
    op.create_index(op.f("ix_tags_user_id"), "tags", ["user_id"], unique=False)
    op.create_table(
        "document_tags",
        sa.Column("document_id", sa.String(length=255), nullable=False),
        sa.Column("tag_id", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_tags_document_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tag_id"], ["tags.id"], name="fk_document_tags_tag_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("document_id", "tag_id", name="pk_document_tags"),
    )


def downgrade() -> None:
    op.drop_table("document_tags")
    op.drop_index(op.f("ix_tags_user_id"), table_name="tags")
    op.drop_table("tags")
    op.drop_index(op.f("ix_documents_user_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_table("documents")
    op.drop_column("app_configs", "notebook_config")
