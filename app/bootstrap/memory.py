"""长期记忆图谱依赖组装入口。"""

from app.bootstrap.notebook import build_embedding_model, build_task_dispatcher
from app.domain.external.llm import LLM
from app.domain.external.task_dispatcher import TaskDispatcher
from app.domain.services.memory.graph_extractor import MemoryGraphExtractor
from app.infrastructure.external.json_parser.repair_json_parser import RepairJSONParser
from app.infrastructure.repositories.neo4j_memory_graph_repository import (
    Neo4jMemoryGraphRepository,
)
from app.infrastructure.storage.neo4j import get_neo4j
from core.config import get_settings

settings = get_settings()


def build_memory_task_dispatcher() -> TaskDispatcher:
    """组装长期记忆后台任务派发器。"""
    return build_task_dispatcher()


def build_memory_graph_repository() -> Neo4jMemoryGraphRepository:
    """组装 Neo4j 记忆图谱仓储。"""
    return Neo4jMemoryGraphRepository(
        driver=get_neo4j().driver,
        database=settings.neo4j_database,
        embedding_dims=settings.notebook_embedding_dims,
    )


async def build_memory_graph_extractor(
    user_id: str,
    llm: LLM,
) -> MemoryGraphExtractor:
    """按用户组装记忆图谱萃取流水线。"""
    embedding = await build_embedding_model(user_id)
    return MemoryGraphExtractor(
        llm=llm,
        embedding=embedding,
        json_parser=RepairJSONParser(),
        graph_repository=build_memory_graph_repository(),
    )


async def ensure_memory_graph_schema() -> None:
    """初始化 Neo4j 记忆图谱 schema。"""
    await build_memory_graph_repository().ensure_schema()
