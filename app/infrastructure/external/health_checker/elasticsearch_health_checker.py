import logging

from app.domain.external.health_checker import HealthChecker
from app.domain.models.health_status import HealthStatus
from app.infrastructure.storage.elasticsearch import KnowledgeElasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchHealthChecker(HealthChecker):
    """Elasticsearch健康检查器。"""

    def __init__(self, elasticsearch: KnowledgeElasticsearch) -> None:
        self._elasticsearch = elasticsearch

    async def check(self) -> HealthStatus:
        """执行轻量 ping，只检查 ES 服务是否可连接。"""
        try:
            if await self._elasticsearch.client.ping():
                return HealthStatus(service="elasticsearch", status="ok")
            return HealthStatus(
                service="elasticsearch",
                status="error",
                details="Elasticsearch服务Ping失败",
            )
        except Exception as e:
            logger.error(f"Elasticsearch健康检查失败: {str(e)}")
            return HealthStatus(
                service="elasticsearch",
                status="error",
                details=str(e),
            )
