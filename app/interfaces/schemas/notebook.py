from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.knowledge import KnowledgeSearchHit as KnowledgeSearchHit


class NotebookDocumentResponse(BaseModel):
    """知识库文档响应模型，避免暴露底层 file_key 等存储信息。"""

    id: str
    file_name: str
    file_ext: str
    file_size: int
    source_type: str
    source_url: str | None
    status: str
    progress: float
    chunk_num: int
    error_msg: str | None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class NotebookDocumentListResponse(BaseModel):
    """知识库文档分页响应模型。"""

    total: int
    page: int
    page_size: int
    items: list[NotebookDocumentResponse]


class NotebookUrlImportRequest(BaseModel):
    """网页导入请求。"""

    url: str = Field(min_length=1, max_length=2048)


class KnowledgeSearchRequest(BaseModel):
    """知识库检索请求。"""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    tags: list[str] | None = None


class NotebookTagResponse(BaseModel):
    """知识库标签响应模型。"""

    id: str
    name: str
