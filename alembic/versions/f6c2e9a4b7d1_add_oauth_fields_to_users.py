"""add oauth fields to users

Revision ID: f6c2e9a4b7d1
Revises: 2d6b7d9a0c11
Create Date: 2026-06-18 16:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f6c2e9a4b7d1"
down_revision: Union[str, Sequence[str], None] = "2d6b7d9a0c11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("users", sa.Column("oauth_provider", sa.String(length=50), nullable=True))
    op.add_column("users", sa.Column("oauth_subject", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("email", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("avatar_url", sa.String(length=1024), nullable=True))
    op.create_unique_constraint(
        "uq_users_oauth_provider_subject",
        "users",
        ["oauth_provider", "oauth_subject"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "uq_users_oauth_provider_subject",
        "users",
        type_="unique",
    )
    op.drop_column("users", "avatar_url")
    op.drop_column("users", "email")
    op.drop_column("users", "oauth_subject")
    op.drop_column("users", "oauth_provider")
