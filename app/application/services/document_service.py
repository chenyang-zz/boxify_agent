import logging
from pathlib import Path

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.application.services.tag_service import TagService
from app.domain.external.document_storage import DocumentStorage
from app.domain.external.knowledge_search import KnowledgeSearch
from app.domain.external.task_dispatcher import TaskDispatcher
from app.domain.external.web_crawler import WebCrawler
from app.domain.models.document import Document
from app.domain.models.knowledge import KnowledgeSearchHit
from app.domain.repositories.vow import IUnitOfWork
from app.utils.hash import content_digest

logger = logging.getLogger(__name__)

SUPPORTED_DOCUMENT_EXTS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024


class DocumentService:
    """知识库文档应用服务，负责协调存储、任务派发和文档元数据事务。"""

    def __init__(
        self,
        uow_factory,
        user_id: str,
        storage: DocumentStorage,
        task_dispatcher: TaskDispatcher,
        tag_service: TagService,
        knowledge_search: KnowledgeSearch | None = None,
        web_crawler: WebCrawler | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._user_id = user_id
        self._storage = storage
        self._task_dispatcher = task_dispatcher
        self._tag_service = tag_service
        self._knowledge_search = knowledge_search
        self._web_crawler = web_crawler

    async def upload_document(self, file_name: str, content: bytes) -> Document:
        """保存上传文件并创建待解析文档记录。"""
        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_DOCUMENT_EXTS:
            raise BadRequestError(msg=f"不支持的文件类型: {ext}")
        if len(content) > MAX_DOCUMENT_SIZE:
            raise BadRequestError(msg="文件超过 50MB 限制")

        document = Document.from_upload(
            user_id=self._user_id,
            file_name=file_name,
            file_key="",
            file_size=len(content),
        )
        document.file_key = f"notebook/{self._user_id}/{document.id}{ext}"
        # 外部对象存储不放进数据库事务，避免上传耗时占用连接和锁。
        await self._storage.save(document.file_key, content)
        logger.info(
            "Notebook文档上传完成: document_id=%s file_size=%s sha256=%s",
            document.id,
            document.file_size,
            content_digest(content),
        )
        async with self._uow_factory() as uow:
            await uow.document.save(document)
        await self._task_dispatcher.dispatch_parse_document(document.id)
        return document

    async def import_url(self, url: str) -> Document:
        """抓取网页正文后按普通文档流程入库并派发解析任务。"""
        if not self._web_crawler:
            raise BadRequestError(msg="网页导入服务未配置")
        web_page = await self._web_crawler.fetch(url)
        content = web_page.text.encode("utf-8")
        document = Document.from_url(
            user_id=self._user_id,
            title=web_page.title or "网页导入",
            source_url=url,
            file_key="",
            file_size=len(content),
        )
        document.file_key = f"notebook/{self._user_id}/{document.id}.txt"
        # 网页内容先落原文件，再用短事务保存文档记录，和上传路径保持一致。
        await self._storage.save(document.file_key, content)
        async with self._uow_factory() as uow:
            await uow.document.save(document)
        await self._task_dispatcher.dispatch_parse_document(document.id)
        return document

    async def list_documents(
        self, page: int, page_size: int, tag: str | None = None
    ) -> tuple[list[Document], int]:
        """分页返回当前用户的文档，按标签过滤时仍保持用户隔离。"""
        async with self._uow_factory() as uow:
            return await uow.document.list_by_user(self._user_id, page, page_size, tag)

    async def get_document(self, document_id: str) -> Document:
        """读取当前用户可见的单个文档。"""
        async with self._uow_factory() as uow:
            return await self._get_or_404(uow, document_id)

    async def retry_document(self, document_id: str) -> Document:
        """将失败或历史文档重新置为待解析状态并重新派发任务。"""
        async with self._uow_factory() as uow:
            document = await self._get_or_404(uow, document_id)
            document.mark_pending()
            await uow.document.save(document)
        await self._task_dispatcher.dispatch_parse_document(document.id)
        return document

    async def delete_document(self, document_id: str) -> None:
        """删除文档相关的检索 chunk、原文件和数据库记录。"""
        async with self._uow_factory() as uow:
            document = await self._get_or_404(uow, document_id)
        if self._knowledge_search:
            await self._knowledge_search.delete_by_source(self._user_id, document.id)
        try:
            await self._storage.delete(document.file_key)
        except Exception as e:
            logger.warning("删除Notebook原文件失败，继续删除数据库记录: %s", e)
        async with self._uow_factory() as uow:
            await uow.document.delete(document)

    async def search_documents(
        self, query: str, top_k: int, tags: list[str] | None = None
    ) -> list[KnowledgeSearchHit]:
        """调用注入的知识库检索能力，服务层只传递用户边界和查询条件。"""
        if not self._knowledge_search:
            raise BadRequestError(msg="Notebook检索服务未配置")
        return await self._knowledge_search.search(
            user_id=self._user_id,
            query=query,
            top_k=top_k,
            tags=tags,
        )

    async def to_response(self, document: Document) -> dict:
        """组装接口响应，避免路由层直接感知标签查询细节。"""
        tags = await self._tag_service.get_document_tags(document.id)
        return {
            "id": document.id,
            "file_name": document.file_name,
            "file_ext": document.file_ext,
            "file_size": document.file_size,
            "source_type": document.source_type.value,
            "source_url": document.source_url,
            "status": document.status.value,
            "progress": document.progress,
            "chunk_num": document.chunk_num,
            "error_msg": document.error_msg,
            "tags": tags,
            "created_at": document.created_at,
        }

    async def _get_or_404(self, uow: IUnitOfWork, document_id: str) -> Document:
        """在当前用户范围内读取文档，不存在时统一转业务异常。"""
        document = await uow.document.get_by_user(self._user_id, document_id)
        if not document:
            raise NotFoundError(msg="文档不存在")
        return document
