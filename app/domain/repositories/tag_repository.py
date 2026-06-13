from abc import ABC, abstractmethod

from app.domain.models.tag import Tag


class TagRepository(ABC):
    """知识库标签仓储接口，标签始终按用户维度隔离。"""

    @abstractmethod
    async def list_by_user(self, user_id: str) -> list[Tag]:
        """查询用户标签列表。"""
        ...

    @abstractmethod
    async def get_document_tags(self, document_id: str) -> list[str]:
        """查询文档关联的标签名。"""
        ...

    @abstractmethod
    async def get_or_create(self, user_id: str, name: str) -> Tag:
        """获取或创建用户级标签。"""
        ...

    @abstractmethod
    async def set_document_tags(self, document_id: str, tag_ids: list[str]) -> None:
        """覆盖文档和标签的关联关系。"""
        ...
