#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/6/8 12:19
@Author : chenyangzhao542@gmail.com
@File   : session_service.py
"""

import logging
from typing import Callable, Dict, List, Optional, Type

from app.application.errors.exceptions import NotFoundError, ServerRequestsError
from app.domain.external.sandbox import Sandbox
from app.domain.models.file import File
from app.domain.models.session import Session
from app.domain.models.tool_result import ToolResult
from app.domain.repositories.vow import IUnitOfWork
from app.interfaces.schemas.session import FileReadResponse, ShellReadResponse

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        sandbox_cls: Type[Sandbox],
    ) -> None:
        """构造函数，完成会话服务的初始化"""
        self._uow_factory = uow_factory
        self._sandbox_cls = sandbox_cls

    async def create_session(self) -> Session:
        """创建一个空白的新任务会话"""
        logger.info("创建一个空白新任务会话")
        session = Session(title="新对话")
        async with self._uow_factory() as uow:
            await uow.session.save(session)
        logger.info(f"成功创建一个新任务会话: {session.id}")
        return session

    async def get_all_sessions(self) -> List[Session]:
        """获取项目所有任务会话列表"""
        async with self._uow_factory() as uow:
            return await uow.session.get_all()

    async def clear_unread_message_count(self, session_id: str) -> None:
        """清空指定会话的未读消息数"""
        logger.info(f"清除会话[{session_id}]未读消息数")
        async with self._uow_factory() as uow:
            await uow.session.update_unread_message_count(session_id, 0)

    async def delete_session(self, session_id: str) -> None:
        """根据传递的会话id删除任务会话"""
        logger.info(f"正在删除会话，会话id: {session_id}")
        # 检查会话是否存在
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
            if not session:
                logger.error(f"会话[{session_id}]不存在，删除失败")
                raise NotFoundError(f"会话[{session_id}]不存在，删除失败")

            # 根据传递的会话id删除会话
            await uow.session.delete_by_id(session_id)
            logger.info(f"删除会话[{session_id}]成功")

    async def get_session(self, session_id: str) -> Optional[Session]:
        """获取指定会话详情信息"""
        async with self._uow_factory() as uow:
            return await uow.session.get_by_id(session_id)

    async def get_session_files(self, session_id: str) -> List[File]:
        """根据传递的会话id获取指定会话的文件列表信息"""

        logger.info(f"获取指定会话[{session_id}]下的文件列表信息")
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
        if not session:
            raise RuntimeError(f"当前会话下不存在[{session_id}]")

        return session.files

    async def read_file(self, session_id: str, filepath: str) -> FileReadResponse:
        """根据传递的信息查看会话中指定文件的内容"""
        # 检查会话是否存在
        logger.info(f"获取会话[{session_id}]中的文件内容，文件路径: {filepath}")
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
        if not session:
            raise RuntimeError(f"当前会话不存在[{session_id}]")

        # 根据沙箱id获取沙箱并判断是否存在
        if not session.sandbox_id:
            raise NotFoundError("当前会话无沙箱环境")
        sandbox = await self._sandbox_cls.get(session.sandbox_id)
        if not sandbox:
            raise NotFoundError("当前会话沙箱不存在或已销毁")

        # 调用沙箱读取文件内容
        result: ToolResult[Dict] = await sandbox.read_file(filepath)
        if result.success and result.data:
            return FileReadResponse(**result.data)

        raise ServerRequestsError(result.message or "")

    async def read_shell_output(
        self, session_id: str, shell_session_id: str
    ) -> ShellReadResponse:
        """根据传递的任务会话id和shell会话id获取shell执行结果"""
        # 检查会话是否存在
        logger.info(
            f"获取会话[{session_id}]中的shell内容输出，shell标识符: {shell_session_id}"
        )
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
        if not session:
            raise RuntimeError(f"当前会话不存在[{session_id}]")

        # 根据沙箱id获取沙箱并判断是否存在
        if not session.sandbox_id:
            raise NotFoundError("当前会话无沙箱环境")
        sandbox = await self._sandbox_cls.get(session.sandbox_id)
        if not sandbox:
            raise NotFoundError("当前会话沙箱不存在或已销毁")

        # 调用沙箱查看shell内容
        result: ToolResult[Dict] = await sandbox.read_shell_output(
            session_id=shell_session_id,
            console=True,
        )
        if result.success and result.data:
            return ShellReadResponse(**result.data)

        raise ServerRequestsError(result.message or "")

    async def get_vnc_url(self, session_id: str) -> str:
        """获取指定会话的vnc连接"""
        # 检查会话是否存在
        logger.info(f"获取会话[{session_id}]的VNC连接")
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)

        if not session:
            raise RuntimeError(f"当前会话不存在[{session_id}]")

        # 根据沙箱id获取沙箱并判断是否存在
        if not session.sandbox_id:
            raise NotFoundError("当前会话无沙箱环境")
        sandbox = await self._sandbox_cls.get(session.sandbox_id)
        if not sandbox:
            raise NotFoundError("当前会话沙箱不存在或已销毁")

        return sandbox.vnc_url
