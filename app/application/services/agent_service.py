#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/6/8 13:13
@Author : chenyangzhao542@gmail.com
@File   : agent_service.py
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Callable, List, Optional, Type

from pydantic import TypeAdapter

from app.domain.agent_task_runner import AgentTaskRunner
from app.domain.external.file_storage import FileStorage
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.external.task import Task
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.event import (
    DoneEvent,
    ErrorEvent,
    Event,
    MessageEvent,
    WaitEvent,
)
from app.domain.models.file import File
from app.domain.models.session import Session, SessionStatus
from app.domain.repositories.vow import IUnitOfWork

logger = logging.getLogger(__name__)


class AgentService:
    """Boxify智能体服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: LLM,
        agent_config: AgentConfig,
        mcp_config: MCPConfig,
        a2a_config: A2AConfig,
        sandbox_cls: Type[Sandbox],
        task_cls: Type[Task],
        json_parser: JSONParser,
        search_engine: SearchEngine,
        file_storage: FileStorage,
    ) -> None:
        """构造函数，完成Agent服务的初始化"""
        self._uow_factory = uow_factory
        self._llm = llm
        self._agent_config = agent_config
        self._mcp_config = mcp_config
        self._a2a_config = a2a_config
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls
        self._json_parser = json_parser
        self._search_engine = search_engine
        self._file_storage = file_storage
        logger.info("AgentService初始化成功")

    async def _get_task(self, session: Session) -> Optional[Task]:
        """根据传递的任务会话获取任务实例"""

        # 从会话中取出任务id
        task_id = session.task_id
        if not task_id:
            return None

        # 调用任务类的get方法获取对应的任务实例
        return self._task_cls.get(task_id)

    async def _create_task(self, session: Session) -> Task:
        """根据传递的会话创建一个新任务"""

        # 获取沙箱实例
        sandbox = None
        sandbox_id = session.sandbox_id
        if sandbox_id:
            sandbox = await self._sandbox_cls.get(sandbox_id)

        # 判断是否能获取到沙箱(如果没有则创建)
        if not sandbox:
            # 沙箱不存在，创建一个新的(被释放了)
            sandbox = await self._sandbox_cls.create()
            session.sandbox_id = sandbox.id

        # 从沙箱中获取浏览器实例
        browser = await sandbox.get_browser()
        if not browser:
            logger.error(f"获取沙箱[{sandbox.id}]中的浏览器实例失败")
            raise RuntimeError(f"获取沙箱[{sandbox.id}]中的浏览器实例失败")

        # 创建AgentTaskRunner
        task_runner = AgentTaskRunner(
            uow_factory=self._uow_factory,
            llm=self._llm,
            agent_config=self._agent_config,
            mcp_config=self._mcp_config,
            a2a_config=self._a2a_config,
            session_id=session.id,
            file_storage=self._file_storage,
            json_parser=self._json_parser,
            browser=browser,
            search_engine=self._search_engine,
            sandbox=sandbox,
        )

        # 创建任务Task并更新会话中的信息
        task = self._task_cls.create(task_runner=task_runner)
        session.task_id = task.id
        async with self._uow_factory() as uow:
            await uow.session.save(session)

        return task

    async def _safe_update_unread_count(self, session_id: str) -> None:
        """在独立的后台任务中安全地更新未读消息计数

        该方法通过asyncio.create_task()调用，运行在一个全新的asyncio Task中，
        因此不受sse_starlette的anyio cancel scope影响，数据库操作可以正常完成。
        使用uow_factory创建全新的UoW实例，避免与被取消的上下文共享数据库连接。
        """
        try:
            uow = self._uow_factory()
            async with uow:
                await uow.session.update_unread_message_count(session_id, 0)
        except Exception as e:
            logger.warning(f"会话[{session_id}]后台更新未读消息计数失败: {e}")

    async def chat(
        self,
        session_id: str,
        message: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        latest_event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> AsyncGenerator[Event, None]:
        """根据传递的信息调用Agent服务发起对话请求"""
        try:
            # 检查会话是否存在
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)

            if not session:
                logger.error(f"尝试与不存在的任务会话[{session_id}]对话")
                raise RuntimeError("任务会话不存在")

            # 获取对应会话任务
            task = await self._get_task(session)

            # 判断是否传递了message
            if message:
                # 判断会话的状态，如果不是运行中则表示已完成或者空闲中
                if session.status != SessionStatus.RUNNING or task is None:
                    # 不在运行中需要创建一个新的task并启动
                    task = await self._create_task(session)
                    if not task:
                        logger.error(f"会话[{session_id}]创建任务失败")
                        raise RuntimeError(f"会话[{session_id}]创建任务失败")

                # 创建一个人类消息事件
                message_event = MessageEvent(
                    role="user",
                    message=message,
                    attachments=[File(id=file_id) for file_id in attachments]
                    if attachments
                    else [],
                )

                # 将事件添加到任务的输入流中
                event_id = await task.input_stream.put(message_event.model_dump_json())
                message_event.id = event_id
                async with self._uow_factory() as uow:
                    await uow.session.update_latest_message(
                        session_id=session_id,
                        message=message,
                        timestamp=timestamp or datetime.now(),
                    )
                    await uow.session.add_event(session_id, message_event)

                # 执行任务
                await task.invoke()
                logger.info(
                    f"往会话[{session_id}]输入消息队列写入消息: {message[:50]}..."
                )

            # 记录日志展示会话已启动
            logger.info(f"会话[{session_id}]已启动")
            logger.info(f"会话[{session_id}]任务实例: {task.id if task else None}")

            # 从任务的输入流中读取数据
            while task and not task.done:
                # 从输出消息队列中获取数据
                event_id, event_str = await task.output_stream.get(
                    start_id=latest_event_id, block_ms=0
                )
                latest_event_id = event_id
                if event_str is None:
                    logger.error(f"在会话[{session_id}]输出队列中未发现事件内容")
                    continue

                # 是用Pydantic提供的类型适配器将event_str转换成指定类型
                event = TypeAdapter(Event).validate_json(event_str)
                event.id = event_id
                logger.debug(f"从会话[{session_id}]中获取事件: {type(event).__name__}")

                # 将未读消息数量重置为0
                async with self._uow_factory() as uow:
                    await uow.session.update_unread_message_count(
                        session_id=session_id, count=0
                    )

                # 将事件返回并判断事件类型释放为结束类型
                yield event
                if isinstance(event, (DoneEvent, ErrorEvent, WaitEvent)):
                    break

            # 循环外面表示这次任务AI短已结束
            logger.info(f"会话[{session_id}]本轮运行结束")

        except Exception as e:
            # 记录日志并返回错误事件
            logger.error(f"任务会话[{session_id}]对话出错: {str(e)}")
            event = ErrorEvent(error=str(e))
            try:
                async with self._uow_factory() as uow:
                    await uow.session.add_event(session_id, event)
            except (asyncio.CancelledError, Exception) as add_err:
                logger.warning(
                    f"会话[{session_id}]添加错误事件失败(可能是客户端断开连接): {add_err}"
                )

            yield event
        finally:
            # 会话完整传递给前端后，表示至少用户肯定收到了这些消息，所以不应该有未读消息数
            # 注意：当SSE客户端断开连接时，sse_starlette使用anyio cancel scope取消当前Task中
            # 所有的await操作（asyncio.shield也无法对抗anyio的cancel scope）。
            # 如果在finally块中直接执行数据库操作，该操作会被立即取消，并且SQLAlchemy在尝试
            # 终止被中断的连接时也会被取消，从而产生ERROR日志并可能污染连接池。
            # 解决方案：将数据库更新操作放到独立的asyncio Task中执行，新Task不受当前
            # cancel scope的影响，可以正常完成数据库操作。
            try:
                asyncio.create_task(self._safe_update_unread_count(session_id))
            except RuntimeError:
                # 事件循环已关闭（如应用正在关闭），无法创建后台任务
                logger.warning(f"会话[{session_id}]无法创建后台任务更新未读消息计数")

    async def stop_session(self, session_id: str) -> None:
        """根据传递的会话id停止指定会话"""

        # 查找会话是否存在
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)

        if not session:
            logger.error(f"尝试获取不存在的会话[{session_id}]")
            raise RuntimeError("任务会话不存在")

        # 根据会话获取任务信息
        task = await self._get_task(session)
        if task:
            task.cancel()

        # 更新会话任务状态
        async with self._uow_factory() as uow:
            await uow.session.update_status(
                session_id,
                SessionStatus.COMPLETED,
            )

    async def shutdown(self) -> None:
        """关闭Agent服务"""
        logger.info("正在清除所有会话任务资源并释放")
        await self._task_cls.destroy()
        logger.info("所有会话记录资源清除成功")
