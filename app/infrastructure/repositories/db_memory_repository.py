from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.long_term_memory import LongTermMemory, MemoryStatus
from app.domain.repositories.memory_repository import MemoryRepository
from app.infrastructure.models.memory import MemoryModel


class DBMemoryRepository(MemoryRepository):
    """基于数据库的长期记忆仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def save(self, memory: LongTermMemory) -> None:
        """保存记忆条目，存在则更新。"""
        record = await self._get_record(memory.id)
        if not record:
            self.db_session.add(MemoryModel.from_domain(memory))
            await self.db_session.flush()
            return
        record.update_from_domain(memory)
        await self.db_session.flush()

    async def get_by_user(self, user_id: str, memory_id: str) -> LongTermMemory | None:
        """按用户和记忆 ID 读取。"""
        stmt = select(MemoryModel).where(
            MemoryModel.user_id == user_id,
            MemoryModel.id == memory_id,
        )
        record = (await self.db_session.execute(stmt)).scalar_one_or_none()
        return record.to_domain() if record else None

    async def list_by_user(
        self, user_id: str, page: int, page_size: int
    ) -> tuple[list[LongTermMemory], int]:
        """分页列出用户记忆。"""
        stmt = (
            select(MemoryModel)
            .where(MemoryModel.user_id == user_id)
            .order_by(MemoryModel.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        count_stmt = select(func.count(MemoryModel.id)).where(
            MemoryModel.user_id == user_id
        )
        records = (await self.db_session.execute(stmt)).scalars().all()
        total = (await self.db_session.execute(count_stmt)).scalar_one()
        return [record.to_domain() for record in records], total

    async def search_by_user(
        self, user_id: str, query: str, top_k: int
    ) -> list[LongTermMemory]:
        """使用当前 PG 文本字段做 V1 检索，后续可替换为图谱检索。"""
        pattern = f"%{query}%"
        stmt = (
            select(MemoryModel)
            .where(
                MemoryModel.user_id == user_id,
                MemoryModel.status == MemoryStatus.COMPLETED,
                or_(
                    MemoryModel.content.ilike(pattern),
                    MemoryModel.summary.ilike(pattern),
                ),
            )
            .order_by(MemoryModel.created_at.desc())
            .limit(top_k)
        )
        records = (await self.db_session.execute(stmt)).scalars().all()
        return [record.to_domain() for record in records]

    async def delete_by_user(self, user_id: str, memory_id: str) -> bool:
        """删除用户记忆。"""
        stmt = delete(MemoryModel).where(
            MemoryModel.user_id == user_id,
            MemoryModel.id == memory_id,
        )
        result = await self.db_session.execute(stmt)
        return bool(result.rowcount)

    async def _get_record(self, memory_id: str) -> MemoryModel | None:
        stmt = select(MemoryModel).where(MemoryModel.id == memory_id)
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()
