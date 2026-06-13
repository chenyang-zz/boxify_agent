from sqlalchemy import func, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.document import Document
from app.domain.repositories.document_repository import DocumentRepository
from app.infrastructure.models.document import DocumentModel
from app.infrastructure.models.document_tag import DocumentTagModel
from app.infrastructure.models.tag import TagModel


class DBDocumentRepository(DocumentRepository):
    """基于数据库的知识库文档仓储，复用 UoW 提供的同一个 session。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def save(self, document: Document) -> None:
        """保存文档领域对象；存在则更新，不存在则插入。"""
        record = await self._get_record(document.id)
        if not record:
            self.db_session.add(DocumentModel.from_domain(document))
            await self.db_session.flush()
            return
        record.update_from_domain(document)
        await self.db_session.flush()

    async def get_by_id(self, document_id: str) -> Document | None:
        """按主键读取文档，供后台任务这类系统流程使用。"""
        record = await self._get_record(document_id)
        return record.to_domain() if record else None

    async def get_by_user(self, user_id: str, document_id: str) -> Document | None:
        """按用户和文档 ID 读取，接口侧所有详情查询都走这个边界。"""
        stmt = select(DocumentModel).where(
            DocumentModel.user_id == user_id,
            DocumentModel.id == document_id,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def list_by_user(
        self, user_id: str, page: int, page_size: int, tag: str | None = None
    ) -> tuple[list[Document], int]:
        """分页查询用户文档，标签过滤时同时约束标签归属用户。"""
        stmt = select(DocumentModel).where(DocumentModel.user_id == user_id)
        count_stmt = select(func.count(DocumentModel.id)).where(
            DocumentModel.user_id == user_id
        )
        if tag:
            stmt = stmt.join(
                DocumentTagModel, DocumentTagModel.document_id == DocumentModel.id
            ).join(TagModel, TagModel.id == DocumentTagModel.tag_id)
            stmt = stmt.where(TagModel.user_id == user_id, TagModel.name == tag)
            count_stmt = (
                select(func.count(DocumentModel.id))
                .join(
                    DocumentTagModel,
                    DocumentTagModel.document_id == DocumentModel.id,
                )
                .join(TagModel, TagModel.id == DocumentTagModel.tag_id)
                .where(DocumentModel.user_id == user_id, TagModel.name == tag)
            )
        stmt = (
            stmt.order_by(DocumentModel.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        records = (await self.db_session.execute(stmt)).scalars().all()
        total = (await self.db_session.execute(count_stmt)).scalar_one()
        return [record.to_domain() for record in records], total

    async def delete(self, document: Document) -> None:
        """删除文档元数据，关联标签关系由数据库外键级联删除。"""
        stmt = delete(DocumentModel).where(DocumentModel.id == document.id)
        await self.db_session.execute(stmt)

    async def _get_record(self, document_id: str) -> DocumentModel | None:
        """读取 ORM 记录，供仓储内部复用。"""
        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()
