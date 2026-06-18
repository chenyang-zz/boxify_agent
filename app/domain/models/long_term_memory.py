from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.models.memory_graph import (
    LongTermMemoryGraphData,
    MemoryGraphStats,
    MemoryTraceResult,
)


class MemorySource(str, Enum):
    """记忆来源。"""

    MANUAL = "manual"
    SESSION = "session"


class MemoryStatus(str, Enum):
    """记忆处理状态。"""

    PENDING = "pending"
    EXTRACTING = "extracting"
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
    graph_dialogue_id: str | None = None
    graph_stats: MemoryGraphStats = Field(default_factory=MemoryGraphStats)
    graph_data: LongTermMemoryGraphData | None = None
    error_msg: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def mark_extracting(self) -> None:
        """标记记忆正在异步萃取图谱。"""
        self.status = MemoryStatus.EXTRACTING
        self.error_msg = None

    def mark_completed(
        self,
        summary: str | None = None,
        keywords: list[str] | None = None,
        graph_dialogue_id: str | None = None,
        graph_stats: MemoryGraphStats | None = None,
    ) -> None:
        """标记记忆图谱萃取完成。"""
        self.status = MemoryStatus.COMPLETED
        self.summary = summary or self.content
        self.keywords = keywords or []
        self.graph_dialogue_id = graph_dialogue_id or self.graph_dialogue_id
        if graph_stats is not None:
            self.graph_stats = graph_stats
        self.error_msg = None

    def mark_failed(self, error_msg: str) -> None:
        """标记记忆处理失败。"""
        self.status = MemoryStatus.FAILED
        self.error_msg = error_msg[:500]


class MemoryDetailResult(LongTermMemory):
    """单条长期记忆详情，包含 PG 记录和可选图谱溯源。"""

    graph_available: bool = True
    trace: MemoryTraceResult | None = None
