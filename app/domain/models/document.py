from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """知识库文档解析状态"""

    PENDING = "pending"
    PARSING = "parsing"
    DONE = "done"
    FAILED = "failed"


class DocumentSourceType(str, Enum):
    """知识库文档来源类型"""

    FILE = "file"
    URL = "url"


class Document(BaseModel):
    """知识库文档领域模型，封装解析状态流转。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    file_name: str
    file_key: str
    file_ext: str = ""
    file_size: int = 0
    source_type: DocumentSourceType = DocumentSourceType.FILE
    source_url: str | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    progress: float = 0
    chunk_num: int = 0
    error_msg: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_upload(
        cls,
        user_id: str,
        file_name: str,
        file_key: str,
        file_size: int,
    ) -> "Document":
        """从上传文件创建文档，扩展名直接来自原始文件名。"""
        return cls(
            user_id=user_id,
            file_name=file_name,
            file_key=file_key,
            file_ext=Path(file_name).suffix.lower(),
            file_size=file_size,
            source_type=DocumentSourceType.FILE,
        )

    @classmethod
    def from_url(
        cls,
        user_id: str,
        title: str,
        source_url: str,
        file_key: str,
        file_size: int,
    ) -> "Document":
        """从网页导入结果创建文档，统一保存为 txt 原文件。"""
        return cls(
            user_id=user_id,
            file_name=f"{title}.txt",
            file_key=file_key,
            file_ext=".txt",
            file_size=file_size,
            source_type=DocumentSourceType.URL,
            source_url=source_url,
        )

    def mark_pending(self) -> None:
        """重置为待解析状态，通常用于重试。"""
        self.status = DocumentStatus.PENDING
        self.progress = 0
        self.error_msg = None

    def mark_parsing(self) -> None:
        """标记解析开始，并给前端一个非零进度。"""
        self.status = DocumentStatus.PARSING
        self.progress = 0.1
        self.error_msg = None

    def mark_done(self, chunk_num: int) -> None:
        """标记解析完成并记录可检索子块数量。"""
        self.status = DocumentStatus.DONE
        self.progress = 1
        self.chunk_num = chunk_num
        self.error_msg = None

    def mark_failed(self, error_msg: str) -> None:
        """标记解析失败，错误信息截断保存以避免响应过大。"""
        self.status = DocumentStatus.FAILED
        self.error_msg = error_msg[:500]
