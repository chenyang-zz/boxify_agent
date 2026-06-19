"""add pinned flags to sessions and projects

Revision ID: 8b2f4d6a9c31
Revises: 3a7e5c9d2b10
Create Date: 2026-06-19 01:35:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "8b2f4d6a9c31"
down_revision: Union[str, Sequence[str], None] = "3a7e5c9d2b10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_projects",
        sa.Column(
            "is_pinned",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "is_pinned",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "is_pinned")
    op.drop_column("session_projects", "is_pinned")
