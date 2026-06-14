from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class MemorySource(str, Enum):
    """记忆来源。"""

    MANUAL = "manual"
    SESSION = "session"


class MemoryStatus(str, Enum):
    """记忆处理状态。"""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class LongTermMemory(BaseModel):
    """用户长期记忆，V1 先保存可检索的原文与轻量摘要。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    content: str
    source: MemorySource = MemorySource.MANUAL
    source_session_id: str | None = None
    status: MemoryStatus = MemoryStatus.PENDING
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    error_msg: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def mark_completed(
        self,
        summary: str | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        """标记记忆可用，V1 的主动记住同步完成。"""
        self.status = MemoryStatus.COMPLETED
        self.summary = summary or self.content
        self.keywords = keywords or []
        self.error_msg = None

    def mark_failed(self, error_msg: str) -> None:
        """标记记忆处理失败。"""
        self.status = MemoryStatus.FAILED
        self.error_msg = error_msg[:500]
