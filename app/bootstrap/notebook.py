"""Notebook 模块依赖组装入口。

FastAPI 依赖和 Celery worker 都通过这里获取领域协议实现，避免在接口层
或任务层散落基础设施实现。
"""

from app.application.services.app_config_service import AppConfigService
from app.domain.external.document_storage import DocumentStorage
from app.domain.external.embedding import EmbeddingModel
from app.domain.external.knowledge_search import KnowledgeSearch
from app.domain.external.task_dispatcher import TaskDispatcher
from app.domain.external.web_crawler import WebCrawler
from app.infrastructure.external.document_storage import CosDocumentStorage
from app.infrastructure.external.embedding import OpenAIEmbedding
from app.infrastructure.external.knowledge_search import ESKnowledgeSearch
from app.infrastructure.external.task_dispatcher import CeleryTaskDispatcher
from app.infrastructure.external.web_crawler import HttpWebCrawler
from app.infrastructure.storage.cos import Cos, get_cos
from app.infrastructure.storage.postgres import get_uow
from core.config import get_settings

settings = get_settings()


class _IndexOnlyEmbedding:
    """仅用于构造索引初始化服务，实际不会发起 embedding 调用。"""

    @property
    def model_name(self) -> str:
        return "index-only"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("索引初始化不应调用 embedding")

    async def embed_one(self, text: str) -> list[float]:
        raise RuntimeError("索引初始化不应调用 embedding")


def build_document_storage(cos: Cos | None = None) -> DocumentStorage:
    """组装知识库文档原文件存储。"""
    return CosDocumentStorage(cos=cos or get_cos(), bucket=settings.cos_bucket)


def build_task_dispatcher() -> TaskDispatcher:
    """组装知识库后台任务派发器。"""
    return CeleryTaskDispatcher()


def build_web_crawler() -> WebCrawler:
    """组装网页正文抓取器。"""
    return HttpWebCrawler()


async def build_embedding_model(user_id: str) -> EmbeddingModel:
    """按用户独立配置组装 Embedding 模型。"""
    app_config = await AppConfigService(
        uow_factory=get_uow,
        user_id=user_id,
    ).get_app_config()
    return OpenAIEmbedding(config=app_config.notebook_config.embedding_config)


async def build_knowledge_search(user_id: str) -> KnowledgeSearch:
    """按用户独立配置组装知识库检索能力。"""
    embedding = await build_embedding_model(user_id)
    return ESKnowledgeSearch(uow_factory=get_uow, embedding=embedding)


async def ensure_knowledge_index() -> None:
    """初始化知识库检索索引，不依赖请求态用户配置。"""
    embedding = _IndexOnlyEmbedding()
    search = ESKnowledgeSearch(uow_factory=get_uow, embedding=embedding)
    await search.ensure_index()
