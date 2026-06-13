from abc import ABC, abstractmethod

from app.domain.models.document import Document


class DocumentRepository(ABC):
    """知识库文档仓储接口，定义应用层所需的持久化能力。"""

    @abstractmethod
    async def save(self, document: Document) -> None:
        """保存文档领域对象。"""
        ...

    @abstractmethod
    async def get_by_id(self, document_id: str) -> Document | None:
        """按文档 ID 读取，主要供后台任务使用。"""
        ...

    @abstractmethod
    async def get_by_user(self, user_id: str, document_id: str) -> Document | None:
        """在用户边界内读取文档。"""
        ...

    @abstractmethod
    async def list_by_user(
        self, user_id: str, page: int, page_size: int, tag: str | None = None
    ) -> tuple[list[Document], int]:
        """分页查询用户文档，可按标签名过滤。"""
        ...

    @abstractmethod
    async def delete(self, document: Document) -> None:
        """删除文档领域对象对应的持久化记录。"""
        ...
