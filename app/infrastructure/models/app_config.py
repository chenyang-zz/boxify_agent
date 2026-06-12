from datetime import datetime
from typing import Any, Dict, Self

from sqlalchemy import DateTime, ForeignKeyConstraint, PrimaryKeyConstraint, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.app_config import AppConfig

from .base import Base


class AppConfigModel(Base):
    """用户应用配置ORM模型"""

    __tablename__ = "app_configs"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", name="pk_app_configs_user_id"),
        ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_app_configs_user_id", ondelete="CASCADE"
        ),
    )

    user_id: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True)
    llm_config: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    agent_config: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    mcp_config: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    a2a_config: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
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
    def from_domain(
        cls,
        user_id: str,
        app_config: AppConfig,
        encryption: Any | None = None,
    ) -> Self:
        """从领域模型构建ORM模型"""
        app_config_data = (
            encryption.encrypt_app_config(app_config)
            if encryption
            else app_config.model_dump(mode="json")
        )
        return cls(user_id=user_id, **app_config_data)

    def to_domain(self, encryption: Any | None = None) -> AppConfig:
        """转换为领域模型"""
        app_config_data = {
            "llm_config": self.llm_config,
            "agent_config": self.agent_config,
            "mcp_config": self.mcp_config,
            "a2a_config": self.a2a_config,
        }
        if encryption:
            app_config_data = encryption.decrypt_app_config_data(app_config_data)
        return AppConfig.model_validate(app_config_data)

    def update_from_domain(
        self,
        app_config: AppConfig,
        encryption: Any | None = None,
    ) -> None:
        """用领域模型更新ORM模型"""
        app_config_data = (
            encryption.encrypt_app_config(app_config)
            if encryption
            else app_config.model_dump(mode="json")
        )
        for field, value in app_config_data.items():
            setattr(self, field, value)
