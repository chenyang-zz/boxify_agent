"""create memories table

Revision ID: 8b1d2f7c9a04
Revises: 4f9d2b7a6c10
Create Date: 2026-06-13 20:20:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "8b1d2f7c9a04"
down_revision: Union[str, Sequence[str], None] = "4f9d2b7a6c10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "memories",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("source_session_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "keywords",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
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
            ["user_id"], ["users.id"], name="fk_memories_user_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_memories_id"),
    )
    op.create_index(op.f("ix_memories_status"), "memories", ["status"], unique=False)
    op.create_index(op.f("ix_memories_user_id"), "memories", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_memories_user_id"), table_name="memories")
    op.drop_index(op.f("ix_memories_status"), table_name="memories")
    op.drop_table("memories")
