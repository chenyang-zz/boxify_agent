from starlette.concurrency import run_in_threadpool

from app.domain.external.document_storage import DocumentStorage
from app.infrastructure.storage.cos import Cos


class CosDocumentStorage(DocumentStorage):
    """基于 COS 的知识库文档原文件存储适配器。"""

    def __init__(self, cos: Cos, bucket: str) -> None:
        self._cos = cos
        self._bucket = bucket

    async def save(self, key: str, content: bytes) -> None:
        """通过线程池调用同步 COS SDK，避免阻塞事件循环。"""
        await run_in_threadpool(
            self._cos.client.put_object,
            Bucket=self._bucket,
            Body=content,
            Key=key,
        )

    async def get(self, key: str) -> bytes:
        """读取 COS 对象内容并返回完整字节流。"""
        response = await run_in_threadpool(
            self._cos.client.get_object,
            Bucket=self._bucket,
            Key=key,
        )
        body = response["Body"]
        return await run_in_threadpool(body.read)

    async def delete(self, key: str) -> None:
        """删除 COS 对象；幂等容错由上层业务服务处理。"""
        await run_in_threadpool(
            self._cos.client.delete_object,
            Bucket=self._bucket,
            Key=key,
        )
