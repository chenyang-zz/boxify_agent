from typing import Protocol, runtime_checkable

from app.domain.models.knowledge import KnowledgeChunk, KnowledgeSearchHit


@runtime_checkable
class KnowledgeSearch(Protocol):
    """知识库检索能力协议"""

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[KnowledgeSearchHit]:
        """在指定用户范围内执行知识库检索。"""
        ...

    async def save_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """保存知识库 chunk 到底层检索索引。"""
        ...

    async def delete_by_source(self, user_id: str, document_id: str) -> None:
        """删除指定用户某个文档对应的检索索引数据。"""
        ...

    async def ensure_index(self) -> None:
        """确保底层检索索引存在。"""
        ...
