from functools import lru_cache

from core.config import get_settings


class KnowledgeElasticsearch:
    """知识库 Elasticsearch 客户端生命周期管理。"""

    def __init__(self) -> None:
        self._client = None
        self._settings = get_settings()

    async def init(self) -> None:
        """按配置初始化 AsyncElasticsearch，重复调用保持幂等。"""
        if self._client is not None:
            return
        from elasticsearch import AsyncElasticsearch

        kwargs = {
            "hosts": [self._settings.elasticsearch_url],
            "request_timeout": self._settings.elasticsearch_request_timeout,
        }
        if self._settings.elasticsearch_username:
            kwargs["basic_auth"] = (
                self._settings.elasticsearch_username,
                self._settings.elasticsearch_password,
            )
        self._client = AsyncElasticsearch(**kwargs)

    async def shutdown(self) -> None:
        """关闭连接并清理缓存，避免 worker 或应用重启后复用旧客户端。"""
        if self._client is not None:
            await self._client.close()
            self._client = None
        get_elasticsearch.cache_clear()

    @property
    def client(self):
        """返回已初始化客户端，未初始化时显式失败。"""
        if self._client is None:
            raise RuntimeError("知识库Elasticsearch未初始化")
        return self._client


@lru_cache()
def get_elasticsearch() -> KnowledgeElasticsearch:
    """提供进程内共享的 Elasticsearch 生命周期对象。"""
    return KnowledgeElasticsearch()
