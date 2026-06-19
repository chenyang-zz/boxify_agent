#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/6/8 16:22
@Author : chenyangzhao542@gmail.com
@File   : db_uow.py
"""

import asyncio
import logging
from typing import Optional, Self, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.repositories.vow import IUnitOfWork
from app.infrastructure.repositories.db_app_config_repository import (
    DBAppConfigRepository,
)
from app.infrastructure.repositories.db_document_repository import DBDocumentRepository
from app.infrastructure.repositories.db_file_repository import DBFileRepository
from app.infrastructure.repositories.db_memory_repository import DBMemoryRepository
from app.infrastructure.repositories.db_session_repository import DBSessionRepository
from app.infrastructure.repositories.db_session_project_repository import (
    DBSessionProjectRepository,
)
from app.infrastructure.repositories.db_tag_repository import DBTagRepository
from app.infrastructure.repositories.db_user_repository import DBUserRepository
from core.config import get_settings

logger = logging.getLogger(__name__)


class DBUnitOfWork(IUnitOfWork):
    """基于Postgres数据库的UoW实例"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """构造函数，完成UoW类初始化"""
        self.session_factory = session_factory
        self.db_session: Optional[AsyncSession] = None
        self._settings = get_settings()

    async def commit(self):
        """提交数据库持久化"""
        if not self.db_session:
            raise RuntimeError("请在异步上下文中执行操作")

        await self.db_session.commit()

    async def rollback(self):
        """数据库回退操作"""
        if not self.db_session:
            raise RuntimeError("请在异步上下文中执行操作")
        await self.db_session.rollback()

    @classmethod
    def _current_task_is_cancelling(cls) -> bool:
        """判断当前 asyncio task 是否正处于取消流程。"""
        task = asyncio.current_task()
        return bool(task and task.cancelling())

    async def _close_session(self, db_session: AsyncSession) -> None:
        """关闭指定 session，并在仍为当前会话时清空引用。"""
        try:
            await db_session.close()
        finally:
            if self.db_session is db_session:
                self.db_session = None

    @classmethod
    async def _rollback_and_close_detached(cls, db_session: AsyncSession) -> None:
        """在后台任务中回滚并关闭已脱离当前 UoW 的 session。"""
        try:
            try:
                await db_session.rollback()
            except asyncio.CancelledError:
                logger.warning("后台UoW回滚操作被取消")
            except Exception as e:
                logger.warning(f"后台UoW回滚操作失败: {e}")
        finally:
            try:
                await db_session.close()
            except asyncio.CancelledError:
                logger.warning("后台UoW关闭会话操作被取消")
            except Exception as e:
                logger.warning(f"后台UoW关闭会话操作失败: {e}")

    def _schedule_rollback_and_close(self, db_session: AsyncSession) -> None:
        """将取消场景下的回滚关闭操作转入后台执行。"""
        cleanup = self._rollback_and_close_detached(db_session)
        try:
            asyncio.create_task(cleanup)
        except RuntimeError:
            cleanup.close()
            logger.warning("无法创建后台UoW清理任务，事件循环可能已关闭")
        finally:
            if self.db_session is db_session:
                self.db_session = None

    async def __aenter__(self) -> Self:
        """进入UoW操作上下文管理器的逻辑"""
        # 为每个上下文开启一个新的会话
        db_session = self.session_factory()
        self.db_session = db_session

        # 初始化所有数据库仓库
        self.app_config = DBAppConfigRepository(
            db_session=db_session,
            encryption_key=self._settings.app_config_encryption_key,
        )
        self.document = DBDocumentRepository(db_session=db_session)
        self.tag = DBTagRepository(db_session=db_session)
        self.file = DBFileRepository(db_session=db_session)
        self.session = DBSessionRepository(db_session=db_session)
        self.session_project = DBSessionProjectRepository(db_session=db_session)
        self.user = DBUserRepository(db_session=db_session)
        self.memory = DBMemoryRepository(db_session=db_session)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时执行的逻辑，如果出现异常则回滚，否则提交

        当SSE客户端断开连接时，sse_starlette的cancel scope会取消所有await操作，
        包括此处的commit/rollback/close。如果不妥善处理CancelledError，
        会导致连接池中的连接处于异常状态，影响后续使用该池的其他任务。
        """
        db_session: AsyncSession = cast(AsyncSession, self.db_session)

        if (
            exc_type
            and issubclass(exc_type, asyncio.CancelledError)
            and self._current_task_is_cancelling()
        ):
            logger.warning("UoW上下文被取消，后台回滚并关闭会话")
            self._schedule_rollback_and_close(db_session)
            return None

        try:
            if exc_type:
                await self.rollback()
            else:
                await self.commit()
        except asyncio.CancelledError:
            logger.warning("UoW提交/回滚操作被取消(可能是客户端断开连接)")
            if self._current_task_is_cancelling():
                self._schedule_rollback_and_close(db_session)
            else:
                await self._close_session(db_session)
            raise
        except Exception as e:
            logger.warning(f"UoW提交/回滚操作失败: {e}")
            raise
        finally:
            if self.db_session is db_session:
                await self._close_session(db_session)
