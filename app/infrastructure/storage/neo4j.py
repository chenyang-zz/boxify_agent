import logging
from functools import lru_cache

from neo4j import AsyncDriver, AsyncGraphDatabase

from core.config import get_settings

logger = logging.getLogger(__name__)


class Neo4j:
    """Neo4j 异步 driver 生命周期管理。"""

    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None
        self._settings = get_settings()

    async def init(self) -> None:
        """初始化 Neo4j driver。"""
        if self._driver is not None:
            logger.warning("Neo4j driver 已初始化，无需重复操作")
            return

        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_username, self._settings.neo4j_password),
        )
        await self._driver.verify_connectivity()
        logger.info("成功连接 Neo4j")

    async def shutdown(self) -> None:
        """关闭 Neo4j driver。"""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("成功关闭 Neo4j 连接")
        get_neo4j.cache_clear()

    @property
    def driver(self) -> AsyncDriver:
        """返回已初始化的 Neo4j driver。"""
        if self._driver is None:
            raise RuntimeError("Neo4j未初始化，请先调用init()函数初始化")
        return self._driver


@lru_cache()
def get_neo4j() -> Neo4j:
    """获取 Neo4j 实例。"""
    return Neo4j()
