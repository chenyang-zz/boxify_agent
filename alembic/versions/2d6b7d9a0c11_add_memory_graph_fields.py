"""add memory graph fields

Revision ID: 2d6b7d9a0c11
Revises: 8b1d2f7c9a04
Create Date: 2026-06-14 14:50:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "2d6b7d9a0c11"
down_revision: Union[str, Sequence[str], None] = "8b1d2f7c9a04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "memories",
        sa.Column("graph_dialogue_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column(
            "graph_stats",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("memories", "graph_stats")
    op.drop_column("memories", "graph_dialogue_id")
