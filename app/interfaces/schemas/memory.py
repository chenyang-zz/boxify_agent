from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.long_term_memory import (
    LongTermMemory,
    MemorySource,
    MemoryStatus,
)


class MemoryCreateRequest(BaseModel):
    """主动记住请求。"""

    content: str = Field(min_length=1, max_length=10000)


class MemorySearchRequest(BaseModel):
    """记忆检索请求。"""

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class MemoryConsolidateResponse(BaseModel):
    """手动记忆巩固统计响应。"""

    promoted_entities: int = 0
    promoted_statements: int = 0
    enhanced_profiles: int = 0


class MemoryReflectResponse(BaseModel):
    """手动记忆反思统计响应。"""

    insights: int = 0
    skipped: str | None = None
    error: str | None = None


class MemoryGraphStatsResponse(BaseModel):
    """长期记忆图谱萃取统计响应。"""

    dialogue_id: str = ""
    chunks: int = 0
    statements: int = 0
    entities: int = 0
    relations: int = 0


class MemoryResponse(BaseModel):
    """记忆响应。"""

    id: str
    content: str
    source: MemorySource
    source_session_id: str | None
    status: MemoryStatus
    summary: str | None
    keywords: list[str]
    graph_dialogue_id: str | None
    graph_stats: MemoryGraphStatsResponse
    error_msg: str | None
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_domain(cls, memory: LongTermMemory) -> "MemoryResponse":
        """从领域模型构造响应。"""
        return cls.model_validate(memory.model_dump(mode="python"))
