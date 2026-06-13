from typing import Protocol


class DocumentStorage(Protocol):
    """知识库文档原文件存储协议"""

    async def save(self, key: str, content: bytes) -> None:
        """保存文档原始内容，key 由应用层按用户和文档 ID 生成。"""
        ...

    async def get(self, key: str) -> bytes:
        """读取文档原始内容，供异步解析任务使用。"""
        ...

    async def delete(self, key: str) -> None:
        """删除文档原始内容，删除失败由应用层决定是否继续清理元数据。"""
        ...
