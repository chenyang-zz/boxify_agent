from datetime import datetime
from typing import Self

from sqlalchemy import Boolean, DateTime, PrimaryKeyConstraint, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.user import User
from .base import Base


class UserModel(Base):
    """用户ORM模型"""

    __tablename__ = "users"
    __table_args__ = (PrimaryKeyConstraint("id", name="pk_users_id"),)

    id: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    is_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
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
    def from_domain(cls, user: User) -> Self:
        """从领域模型构建ORM模型"""
        return cls(**user.model_dump(mode="python"))

    def to_domain(self) -> User:
        """转换为领域模型"""
        return User.model_validate(self, from_attributes=True)

    def update_from_domain(self, user: User) -> None:
        """用领域模型更新ORM模型"""
        for field, value in user.model_dump(mode="python").items():
            setattr(self, field, value)
