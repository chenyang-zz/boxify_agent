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


class MemoryResponse(BaseModel):
    """记忆响应。"""

    id: str
    content: str
    source: MemorySource
    source_session_id: str | None
    status: MemoryStatus
    summary: str | None
    keywords: list[str]
    error_msg: str | None
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_domain(cls, memory: LongTermMemory) -> "MemoryResponse":
        """从领域模型构造响应。"""
        return cls.model_validate(memory.model_dump(mode="python"))
