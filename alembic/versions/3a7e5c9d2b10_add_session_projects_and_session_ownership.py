"""add session projects and session ownership

Revision ID: 3a7e5c9d2b10
Revises: f6c2e9a4b7d1
Create Date: 2026-06-19 00:40:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "3a7e5c9d2b10"
down_revision: Union[str, Sequence[str], None] = "f6c2e9a4b7d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "session_projects",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("sort_order", sa.Integer(), server_default=sa.text("0"), nullable=False),
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
            ["user_id"],
            ["users.id"],
            name="fk_session_projects_user_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_session_projects_id"),
        sa.UniqueConstraint("user_id", "name", name="uq_session_projects_user_id_name"),
    )
    op.create_index(
        op.f("ix_session_projects_user_id"),
        "session_projects",
        ["user_id"],
        unique=False,
    )

    op.add_column("sessions", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.add_column(
        "sessions", sa.Column("project_id", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "sessions",
        sa.Column(
            "type",
            sa.String(length=32),
            server_default=sa.text("'task'::character varying"),
            nullable=False,
        ),
    )

    bind = op.get_bind()
    session_count = bind.execute(sa.text("SELECT COUNT(*) FROM sessions")).scalar_one()
    owner_id = bind.execute(
        sa.text(
            """
            SELECT id
            FROM users
            ORDER BY CASE WHEN is_admin THEN 0 ELSE 1 END, created_at ASC
            LIMIT 1
            """
        )
    ).scalar_one_or_none()
    if session_count and not owner_id:
        raise RuntimeError(
            "sessions table contains rows but users table is empty; "
            "create an admin user before applying this migration"
        )
    if owner_id:
        bind.execute(
            sa.text("UPDATE sessions SET user_id = :owner_id WHERE user_id IS NULL"),
            {"owner_id": owner_id},
        )

    op.alter_column("sessions", "user_id", existing_type=sa.String(length=255), nullable=False)
    op.create_index(op.f("ix_sessions_user_id"), "sessions", ["user_id"], unique=False)
    op.create_index(op.f("ix_sessions_project_id"), "sessions", ["project_id"], unique=False)
    op.create_foreign_key(
        "fk_sessions_user_id",
        "sessions",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_sessions_project_id",
        "sessions",
        "session_projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_sessions_project_id", "sessions", type_="foreignkey")
    op.drop_constraint("fk_sessions_user_id", "sessions", type_="foreignkey")
    op.drop_index(op.f("ix_sessions_project_id"), table_name="sessions")
    op.drop_index(op.f("ix_sessions_user_id"), table_name="sessions")
    op.drop_column("sessions", "type")
    op.drop_column("sessions", "project_id")
    op.drop_column("sessions", "user_id")
    op.drop_index(op.f("ix_session_projects_user_id"), table_name="session_projects")
    op.drop_table("session_projects")
