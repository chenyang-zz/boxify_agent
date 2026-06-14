import logging

from app.domain.external.health_checker import HealthChecker
from app.domain.models.health_status import HealthStatus
from app.infrastructure.storage.neo4j import Neo4j

logger = logging.getLogger(__name__)


class Neo4jHealthChecker(HealthChecker):
    """Neo4j 健康检查器。"""

    def __init__(self, neo4j: Neo4j) -> None:
        self._neo4j = neo4j

    async def check(self) -> HealthStatus:
        """检查 Neo4j driver 连通性。"""
        try:
            await self._neo4j.driver.verify_connectivity()
            return HealthStatus(service="neo4j", status="ok")
        except Exception as e:
            logger.error("Neo4j健康检查失败: %s", str(e))
            return HealthStatus(service="neo4j", status="error", details=str(e))
