"""长期记忆图谱依赖组装入口。"""

from app.bootstrap.notebook import build_embedding_model, build_task_dispatcher
from app.domain.external.llm import LLM
from app.domain.external.task_dispatcher import TaskDispatcher
from app.domain.models.app_config import AppConfig
from app.domain.services.memory.fact_extractor import MemoryFactExtractor
from app.domain.services.memory.consolidation import MemoryConsolidator
from app.domain.services.memory.graph_extractor import MemoryGraphExtractor
from app.domain.services.memory.profile_summarizer import MemoryProfileSummarizer
from app.infrastructure.external.json_parser.repair_json_parser import RepairJSONParser
from app.infrastructure.external.llm.openai_llm import OpenAILLM
from app.infrastructure.repositories.neo4j_memory_graph_repository import (
    Neo4jMemoryGraphRepository,
)
from app.infrastructure.storage.neo4j import get_neo4j
from app.infrastructure.storage.postgres import get_uow
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
    fact_extractor = MemoryFactExtractor(llm=llm, json_parser=RepairJSONParser())
    return MemoryGraphExtractor(
        fact_extractor=fact_extractor,
        embedding=embedding,
        graph_repository=build_memory_graph_repository(),
    )


async def build_memory_graph_extractor_for_user(user_id: str) -> MemoryGraphExtractor:
    """按用户应用配置组装记忆图谱萃取流水线。"""
    app_config = await _load_user_app_config(user_id)
    llm = OpenAILLM(app_config.llm_config)
    return await build_memory_graph_extractor(user_id=user_id, llm=llm)


async def build_memory_consolidation_service_for_user(
    user_id: str,
) -> MemoryConsolidator:
    """按用户应用配置组装记忆巩固服务。"""
    app_config = await _load_user_app_config(user_id)
    llm = OpenAILLM(app_config.llm_config)
    return MemoryConsolidator(
        user_id=user_id,
        graph_repository=build_memory_graph_repository(),
        profile_summarizer=MemoryProfileSummarizer(
            llm=llm,
            json_parser=RepairJSONParser(),
        ),
    )


async def _load_user_app_config(user_id: str) -> AppConfig:
    """通过 UoW 读取当前用户应用配置，不依赖应用层服务。"""
    async with get_uow() as uow:
        return await uow.app_config.get_or_create_default(user_id)


async def ensure_memory_graph_schema() -> None:
    """初始化 Neo4j 记忆图谱 schema。"""
    await build_memory_graph_repository().ensure_schema()
