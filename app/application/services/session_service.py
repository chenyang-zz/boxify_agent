#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/6/8 12:19
@Author : chenyangzhao542@gmail.com
@File   : session_service.py
"""

import logging
from typing import Callable, Dict, List, Optional, Type

from app.application.errors.exceptions import BadRequestError, NotFoundError, ServerRequestsError
from app.domain.external.sandbox import Sandbox
from app.domain.models.file import File
from app.domain.models.session import Session, SessionType
from app.domain.models.session_project import SessionProject
from app.domain.models.tool_result import ToolResult
from app.domain.repositories.vow import IUnitOfWork
from app.interfaces.schemas.session import FileReadResponse, ShellReadResponse

logger = logging.getLogger(__name__)
UNSET_PROJECT_ID = object()


class SessionService:
    """会话服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        sandbox_cls: Type[Sandbox],
        user_id: str = "",
    ) -> None:
        """构造函数，完成会话服务的初始化"""
        self._uow_factory = uow_factory
        self._sandbox_cls = sandbox_cls
        self._user_id = user_id

    async def create_session(
        self,
        session_type: SessionType | str = SessionType.CHAT,
        project_id: str | None = None,
        is_pinned: bool = False,
    ) -> Session:
        """创建一个空白的新任务会话"""
        logger.info("创建一个空白新任务会话")
        normalized_type = self._normalize_session_type(session_type)
        async with self._uow_factory() as uow:
            if project_id:
                await self._ensure_project_exists(uow, project_id)
            session = Session(
                user_id=self._user_id,
                title="新对话",
                type=normalized_type,
                project_id=project_id,
                is_pinned=is_pinned,
            )
            await uow.session.save(session)
        logger.info(f"成功创建一个新任务会话: {session.id}")
        return session

    async def get_all_sessions(self) -> List[Session]:
        """获取项目所有任务会话列表"""
        async with self._uow_factory() as uow:
            if self._user_id:
                return await uow.session.get_all_by_user(self._user_id)
            return await uow.session.get_all()

    async def create_project(
        self, name: str, sort_order: int = 0, is_pinned: bool = False
    ) -> SessionProject:
        """创建当前用户的会话项目。"""
        project_name = name.strip()
        if not project_name:
            raise BadRequestError("项目名称不能为空")
        project = SessionProject(
            user_id=self._user_id,
            name=project_name,
            sort_order=sort_order,
            is_pinned=is_pinned,
        )
        async with self._uow_factory() as uow:
            await uow.session_project.save(project)
        return project

    async def update_project(
        self,
        project_id: str,
        name: str | None = None,
        sort_order: int | None = None,
        is_pinned: bool | None = None,
    ) -> SessionProject:
        """更新当前用户的会话项目。"""
        async with self._uow_factory() as uow:
            project = await uow.session_project.get_by_id_for_user(
                project_id, self._user_id
            )
            if not project:
                raise NotFoundError("项目不存在")
            if name is not None:
                project_name = name.strip()
                if not project_name:
                    raise BadRequestError("项目名称不能为空")
                project.name = project_name
            if sort_order is not None:
                project.sort_order = sort_order
            if is_pinned is not None:
                project.is_pinned = is_pinned
            await uow.session_project.save(project)
        return project

    async def delete_project(self, project_id: str) -> None:
        """删除当前用户项目及其下会话。"""
        async with self._uow_factory() as uow:
            project = await uow.session_project.get_by_id_for_user(
                project_id, self._user_id
            )
            if not project:
                raise NotFoundError("项目不存在")
            await uow.session.delete_by_project(project_id, self._user_id)
            await uow.session_project.delete_by_id_for_user(project_id, self._user_id)

    async def get_sidebar_projects(self) -> List[SessionProject]:
        """获取当前用户侧边栏项目列表。"""
        async with self._uow_factory() as uow:
            return await uow.session_project.list_by_user(self._user_id)

    async def get_sidebar_sessions(self) -> List[Session]:
        """获取当前用户侧边栏会话列表，仅包含普通聊天会话。"""
        async with self._uow_factory() as uow:
            sessions = await uow.session.get_all_by_user(self._user_id)
        return [
            session
            for session in sessions
            if (
                session.type
                if isinstance(session.type, SessionType)
                else SessionType(session.type)
            )
            == SessionType.CHAT
        ]

    async def update_session(
        self,
        session_id: str,
        title: str | None = None,
        project_id: str | None | object = UNSET_PROJECT_ID,
        is_pinned: bool | None = None,
    ) -> Session:
        """更新会话标题或项目归属。"""
        async with self._uow_factory() as uow:
            session = await self._get_session_for_user(uow, session_id)
            if title is not None:
                session.title = title
            if project_id is not UNSET_PROJECT_ID:
                if project_id:
                    await self._ensure_project_exists(uow, project_id)
                session.project_id = project_id
            if is_pinned is not None:
                session.is_pinned = is_pinned
            await uow.session.save(session)
        return session

    async def clear_unread_message_count(self, session_id: str) -> None:
        """清空指定会话的未读消息数"""
        logger.info(f"清除会话[{session_id}]未读消息数")
        async with self._uow_factory() as uow:
            await self._get_session_for_user(uow, session_id)
            await uow.session.update_unread_message_count(session_id, 0)

    async def delete_session(self, session_id: str) -> None:
        """根据传递的会话id删除任务会话"""
        logger.info(f"正在删除会话，会话id: {session_id}")
        # 检查会话是否存在
        async with self._uow_factory() as uow:
            session = await self._get_session_for_user(uow, session_id)
            if not session:
                logger.error(f"会话[{session_id}]不存在，删除失败")
                raise NotFoundError(f"会话[{session_id}]不存在，删除失败")

            # 根据传递的会话id删除会话
            await uow.session.delete_by_id_for_user(session_id, self._user_id)
            logger.info(f"删除会话[{session_id}]成功")

    async def get_session(self, session_id: str) -> Optional[Session]:
        """获取指定会话详情信息"""
        async with self._uow_factory() as uow:
            if self._user_id:
                return await uow.session.get_by_id_for_user(session_id, self._user_id)
            return await uow.session.get_by_id(session_id)

    async def get_session_files(self, session_id: str) -> List[File]:
        """根据传递的会话id获取指定会话的文件列表信息"""

        logger.info(f"获取指定会话[{session_id}]下的文件列表信息")
        async with self._uow_factory() as uow:
            session = await self._get_session_for_user(uow, session_id)
        if not session:
            raise RuntimeError(f"当前会话下不存在[{session_id}]")

        return session.files

    async def read_file(self, session_id: str, filepath: str) -> FileReadResponse:
        """根据传递的信息查看会话中指定文件的内容"""
        # 检查会话是否存在
        logger.info(f"获取会话[{session_id}]中的文件内容，文件路径: {filepath}")
        async with self._uow_factory() as uow:
            session = await self._get_session_for_user(uow, session_id)
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
            session = await self._get_session_for_user(uow, session_id)
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
            session = await self._get_session_for_user(uow, session_id)

        if not session:
            raise RuntimeError(f"当前会话不存在[{session_id}]")

        # 根据沙箱id获取沙箱并判断是否存在
        if not session.sandbox_id:
            raise NotFoundError("当前会话无沙箱环境")
        sandbox = await self._sandbox_cls.get(session.sandbox_id)
        if not sandbox:
            raise NotFoundError("当前会话沙箱不存在或已销毁")

        return sandbox.vnc_url

    async def _ensure_project_exists(self, uow: IUnitOfWork, project_id: str) -> None:
        """校验项目存在且属于当前用户。"""
        project = await uow.session_project.get_by_id_for_user(project_id, self._user_id)
        if not project:
            raise NotFoundError("项目不存在")

    async def _get_session_for_user(
        self, uow: IUnitOfWork, session_id: str
    ) -> Session:
        """按当前用户查询会话，不存在时抛统一错误。"""
        session = (
            await uow.session.get_by_id_for_user(session_id, self._user_id)
            if self._user_id
            else await uow.session.get_by_id(session_id)
        )
        if not session:
            raise NotFoundError("该会话不存在")
        return session

    @staticmethod
    def _normalize_session_type(session_type: SessionType | str) -> SessionType:
        """规范化会话类型。"""
        try:
            return (
                session_type
                if isinstance(session_type, SessionType)
                else SessionType(session_type)
            )
        except ValueError:
            raise BadRequestError("不支持的会话类型") from None
