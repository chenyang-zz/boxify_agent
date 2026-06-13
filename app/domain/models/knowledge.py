from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    """可写入知识库检索索引的 chunk 领域模型。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    source_type: str = "document"
    source_id: str
    doc_name: str
    chunk_type: str
    parent_id: str | None = None
    content: str
    vector: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def chunk_id(self) -> str:
        """兼容检索索引中的 chunk_id 字段。"""
        return self.id


class KnowledgeSearchHit(BaseModel):
    """知识库检索命中结果领域模型。"""

    chunk_id: str
    content: str
    doc_name: str | None = None
    source_id: str | None = None
    source_type: str | None = None
    score: float
