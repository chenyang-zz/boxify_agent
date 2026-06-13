from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.tag import Tag
from app.domain.repositories.tag_repository import TagRepository
from app.infrastructure.models.document_tag import DocumentTagModel
from app.infrastructure.models.tag import TagModel


class DBTagRepository(TagRepository):
    """基于数据库的知识库标签仓储，复用 UoW 提供的同一个 session。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def list_by_user(self, user_id: str) -> list[Tag]:
        """查询用户标签并按名称排序，保证接口响应稳定。"""
        stmt = select(TagModel).where(TagModel.user_id == user_id).order_by(TagModel.name)
        records = (await self.db_session.execute(stmt)).scalars().all()
        return [record.to_domain() for record in records]

    async def get_document_tags(self, document_id: str) -> list[str]:
        """查询文档关联标签名，用于文档响应展示。"""
        stmt = (
            select(TagModel.name)
            .join(DocumentTagModel, DocumentTagModel.tag_id == TagModel.id)
            .where(DocumentTagModel.document_id == document_id)
            .order_by(TagModel.name)
        )
        return list((await self.db_session.execute(stmt)).scalars().all())

    async def get_or_create(self, user_id: str, name: str) -> Tag:
        """按用户维度复用同名标签，不跨用户共享标签。"""
        normalized_name = name.strip()
        stmt = select(TagModel).where(
            TagModel.user_id == user_id,
            TagModel.name == normalized_name,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if record:
            return record.to_domain()
        tag = Tag(user_id=user_id, name=normalized_name)
        self.db_session.add(TagModel.from_domain(tag))
        await self.db_session.flush()
        return tag

    async def set_document_tags(self, document_id: str, tag_ids: list[str]) -> None:
        """重置文档标签关系，调用方负责保证 tag_ids 属于当前用户。"""
        await self.db_session.execute(
            delete(DocumentTagModel).where(DocumentTagModel.document_id == document_id)
        )
        for tag_id in tag_ids:
            self.db_session.add(DocumentTagModel(document_id=document_id, tag_id=tag_id))
        await self.db_session.flush()
