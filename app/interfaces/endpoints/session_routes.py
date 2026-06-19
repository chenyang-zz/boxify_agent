#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/6/8 12:15
@Author : chenyangzhao542@gmail.com
@File   : session_routes.py
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

import websockets
from fastapi import APIRouter, Body, Depends
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.application.services.agent_service import AgentService
from app.application.services.chat_service import ChatService
from app.application.services.session_service import SessionService
from app.domain.models.session import SessionType
from app.interfaces.schemas import Response
from app.interfaces.schemas.event import EventMapper
from app.interfaces.schemas.session import (
    ChatRequest,
    CreateSessionProjectRequest,
    CreateSessionRequest,
    SessionProjectRequest,
    SessionProjectResponse,
    CreateSessionResponse,
    FileReadRequest,
    FileReadResponse,
    GetSessionFilesResponse,
    GetSessionResponse,
    ListSessionItem,
    ListSessionResponse,
    SidebarProjectItem,
    ShellReadRequest,
    ShellReadResponse,
    SessionSidebarResponse,
    UpdateSessionRequest,
)
from app.interfaces.service_dependencies import (
    get_agent_service,
    get_chat_service,
    get_session_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["会话模块"])


# 流式获取会话详情睡眠间隔
SESSION_SLEEP_INTERVAL = 5


def to_list_session_item(session) -> ListSessionItem:
    """将会话领域对象转换为列表条目。"""
    session_type = (
        session.type if isinstance(session.type, SessionType) else SessionType(session.type)
    )
    data = {
        "session_id": session.id,
        "title": session.title,
        "latest_message": session.latest_message,
        "latest_message_at": session.latest_message_at,
        "status": session.status,
        "type": session_type,
        "project_id": session.project_id,
        "is_pinned": session.is_pinned,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }
    if session_type != SessionType.CHAT:
        data["unread_message_count"] = session.unread_message_count
    return ListSessionItem(**data)


def to_project_response(project) -> SessionProjectResponse:
    """将项目领域对象转换为响应结构。"""
    return SessionProjectResponse(
        project_id=project.id,
        name=project.name,
        sort_order=project.sort_order,
        is_pinned=project.is_pinned,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def to_sidebar_response(projects, sessions) -> SessionSidebarResponse:
    """将项目和会话领域对象组装为侧边栏响应结构。"""
    sessions_by_project = {project.id: [] for project in projects}
    standalone = []
    for session in sessions:
        if session.project_id and session.project_id in sessions_by_project:
            sessions_by_project[session.project_id].append(session)
        else:
            standalone.append(session)
    return SessionSidebarResponse(
        projects=[
            SidebarProjectItem(
                **to_project_response(project).model_dump(),
                sessions=[
                    to_list_session_item(session)
                    for session in sessions_by_project[project.id]
                ],
            )
            for project in projects
        ],
        standalone_conversations=[
            to_list_session_item(session) for session in standalone
        ],
    )


@router.post(
    path="",
    summary="创建新任务会话",
    description="创建一个空白的新任务会话",
    response_model=Response[CreateSessionResponse],
)
async def create_session(
    request: CreateSessionRequest = Body(default_factory=CreateSessionRequest),
    session_service: SessionService = Depends(get_session_service),
):
    """创建一个空白的新任务会话"""
    session = await session_service.create_session(
        session_type=request.type.value,
        project_id=request.project_id,
        is_pinned=request.is_pinned,
    )
    return Response.success(
        msg="创建任务会话成功",
        data=CreateSessionResponse(
            session_id=session.id,
            type=session.type,
            project_id=session.project_id,
            is_pinned=session.is_pinned,
        ),
    )


@router.get(
    path="",
    summary="获取会话列表基础信息",
    description="获取Boxify项目中所有任务会话基础信息列表",
    response_model=Response[ListSessionResponse],
    response_model_exclude_unset=True,
)
async def get_all_sessions(
    session_service: SessionService = Depends(get_session_service),
):
    """获取Boxify项目中所有任务会话基础信息列表"""
    sessions = await session_service.get_all_sessions()
    session_items = [to_list_session_item(session) for session in sessions]
    return Response.success(
        msg="获取任务会话列表成功",
        data=ListSessionResponse(sessions=session_items),
    )


@router.get(
    path="/sidebar",
    summary="获取侧边栏会话项目结构",
    description="获取项目、项目下会话和独立会话的组合结构",
    response_model=Response[SessionSidebarResponse],
    response_model_exclude_unset=True,
)
async def get_sidebar(
    session_service: SessionService = Depends(get_session_service),
):
    """获取侧边栏组合结构"""
    projects = await session_service.get_sidebar_projects()
    sessions = await session_service.get_sidebar_sessions()
    return Response.success(
        msg="获取侧边栏会话列表成功",
        data=to_sidebar_response(projects, sessions),
    )


@router.post(
    path="/projects",
    summary="创建会话项目",
    description="创建当前用户的会话项目",
    response_model=Response[SessionProjectResponse],
)
async def create_project(
    request: CreateSessionProjectRequest,
    session_service: SessionService = Depends(get_session_service),
):
    """创建会话项目"""
    project = await session_service.create_project(
        name=request.name,
        sort_order=request.sort_order,
        is_pinned=request.is_pinned,
    )
    return Response.success(msg="创建会话项目成功", data=to_project_response(project))


@router.post(
    path="/projects/{project_id}/update",
    summary="更新会话项目",
    description="更新当前用户的会话项目名称或排序",
    response_model=Response[SessionProjectResponse],
)
async def update_project(
    project_id: str,
    request: SessionProjectRequest,
    session_service: SessionService = Depends(get_session_service),
):
    """更新会话项目"""
    project = await session_service.update_project(
        project_id=project_id,
        name=request.name,
        sort_order=request.sort_order,
        is_pinned=request.is_pinned,
    )
    return Response.success(msg="更新会话项目成功", data=to_project_response(project))


@router.post(
    path="/projects/{project_id}/delete",
    summary="删除会话项目",
    description="删除当前用户的会话项目及项目下会话",
    response_model=Response[Optional[Dict]],
)
async def delete_project(
    project_id: str,
    session_service: SessionService = Depends(get_session_service),
):
    """删除会话项目及项目下会话"""
    await session_service.delete_project(project_id)
    return Response.success(msg="删除会话项目成功")


@router.post(
    path="/{session_id}/update",
    summary="更新指定会话",
    description="更新会话标题或移动到指定项目，project_id为空表示独立会话",
    response_model=Response[ListSessionItem],
    response_model_exclude_unset=True,
)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    session_service: SessionService = Depends(get_session_service),
):
    """更新会话标题或项目归属"""
    update_kwargs = {}
    if request.title is not None:
        update_kwargs["title"] = request.title
    if "project_id" in request.model_fields_set:
        update_kwargs["project_id"] = request.project_id
    if "is_pinned" in request.model_fields_set:
        update_kwargs["is_pinned"] = request.is_pinned
    session = await session_service.update_session(session_id, **update_kwargs)
    return Response.success(msg="更新会话成功", data=to_list_session_item(session))


@router.post(
    path="/{session_id}/clear-unread-message-count",
    summary="清除指定任务会话未读消息数",
    description="根据传递的会话id清空未读消息数",
    response_model=Response[Optional[Dict]],
)
async def clear_unread_message_count(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
):
    """根据传递的会话id清空未读消息数"""
    await session_service.clear_unread_message_count(session_id)
    return Response.success(msg="清除未读消息数成功")


@router.post(
    path="/{session_id}/delete",
    summary="删除指定任务会话",
    description="根据传递的会话id删除指定任务会话",
    response_model=Response[Optional[Dict]],
)
async def delete_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
):
    """根据传递的会话id删除指定任务会话"""
    await session_service.delete_session(session_id)
    return Response.success(msg="删除任务会话成功")


@router.post(
    path="/{session_id}/chat",
    summary="向指定任务会话发起聊天请求",
    description="根据传递的会话id和chat请求数据向指定会话发起聊天请求",
)
async def chat(
    session_id: str,
    request: ChatRequest,
    session_service: SessionService = Depends(get_session_service),
    chat_service: ChatService = Depends(get_chat_service),
    agent_service: AgentService = Depends(get_agent_service),
):
    """根据传递的会话id和chat请求数据向指定会话发起聊天请求"""
    session = await session_service.get_session(session_id)
    if not session:
        raise NotFoundError("该会话不存在")
    session_type = (
        session.type if isinstance(session.type, SessionType) else SessionType(session.type)
    )
    if session_type == SessionType.CHAT and request.attachments:
        raise BadRequestError("普通聊天会话暂不支持附件")

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """定义事件生成器，用于配合EventSourceResponse生成流式响应数据"""
        service = chat_service if session_type == SessionType.CHAT else agent_service
        chat_kwargs = {
            "session_id": session_id,
            "message": request.message,
            "latest_event_id": request.event_id,
            "timestamp": datetime.fromtimestamp(request.timestamp)
            if request.timestamp
            else None,
        }
        if session_type == SessionType.TASK:
            chat_kwargs["attachments"] = request.attachments
        async for event in service.chat(**chat_kwargs):
            # 将Agent事件转换为see数据(因为普通的event没法通过流式事件传输)
            sse_event = EventMapper.event_to_sse_event(event)
            if sse_event:
                yield ServerSentEvent(
                    event=sse_event.event,
                    data=sse_event.data.model_dump_json(),
                )

    return EventSourceResponse(event_generator())


@router.get(
    path="/{session_id}",
    summary="获取指定会话详情信息",
    description="根据传递的会话id获取该会话的对话详情",
    response_model=Response[GetSessionResponse],
)
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
):
    """ "根据传递的会话id获取该会话的对话详情"""

    session = await session_service.get_session(session_id)
    if not session:
        raise NotFoundError("该会话不存在")
    return Response.success(
        msg="获取会话详情成功",
        data=GetSessionResponse(
            session_id=session.id,
            title=session.title,
            status=session.status,
            events=EventMapper.events_to_sse_events(session.events),
        ),
    )


@router.post(
    path="/stream",
    summary="流式获取所有会话基础信息列表",
    description="间隔指定事件流式获取所有会话基础信息列表",
)
async def stream_sessions(
    session_service: SessionService = Depends(get_session_service),
):
    """间隔指定事件流式获取所有会话基础信息列表"""

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """定义一个异步迭代器，用于获取所有会话列表"""

        while True:
            sessions = await session_service.get_all_sessions()
            session_items = [to_list_session_item(session) for session in sessions]

            # 将会话列表转换为流式事件数据并返回
            yield ServerSentEvent(
                event="sessions",
                data=ListSessionResponse(sessions=session_items).model_dump_json(
                    exclude_unset=True
                ),
            )

            # 睡眠指定事件避免高频响应
            await asyncio.sleep(SESSION_SLEEP_INTERVAL)

    return EventSourceResponse(event_generator())


@router.post(
    path="/{session_id}/stop",
    summary="停止指定任务会话",
    description="根据传递的指定会话id停止对应任务会话",
    response_model=Response[Optional[Dict]],
)
async def stop_session(
    session_id: str,
    agent_service: AgentService = Depends(get_agent_service),
):
    """根据传递的指定会话id停止对应任务会话"""

    await agent_service.stop_session(session_id)
    return Response.success(msg="停止任务会话成功")


@router.get(
    path="/{session_id}/files",
    summary="获取指定会话文件列表信息",
    description="获取指定任务会话文件列表信息",
    response_model=Response[GetSessionFilesResponse],
)
async def get_session_files(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
):
    """获取指定任务会话文件列表信息"""

    files = await session_service.get_session_files(session_id)
    return Response.success(
        msg="获取会话文件列表成功",
        data=GetSessionFilesResponse(files=files),
    )


@router.post(
    path="/{session_id}/file",
    summary="查看会话沙箱中指定文件的内容",
    description="根据传递的会话ID和文件路径查看沙箱中文件的内容信息",
    response_model=Response[FileReadResponse],
)
async def read_file(
    session_id: str,
    request: FileReadRequest,
    session_service: SessionService = Depends(get_session_service),
):
    """根据传递的会话ID和文件路径查看沙箱中文件的内容信息"""
    result = await session_service.read_file(session_id, request.filepath)
    return Response.success(
        msg="获取会话文件内容成功",
        data=result,
    )


@router.post(
    path="/{session_id}/shell",
    summary="查看会话的shell内容输出",
    description="传递指定会话id与shell会话表示，查看shell内容输出",
    response_model=Response[ShellReadResponse],
)
async def read_shell_output(
    session_id: str,
    request: ShellReadRequest,
    session_service: SessionService = Depends(get_session_service),
):
    """传递指定会话id与shell会话表示，查看shell内容输出"""
    result = await session_service.read_shell_output(session_id, request.session_id)
    return Response.success(
        msg="获取Shell内容输出结果成功",
        data=result,
    )


@router.websocket(
    path="/{session_id}/vnc",
)
async def vnc_websocket(
    websocket: WebSocket,
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
):
    """VCN Websocket端点，用于建立与沙箱的vnc连接，并双向转发数据"""

    # 从客户端noVNC接收子协议
    protocols_str = websocket.headers.get("sec-websocket-protocol", "")
    protocols = [p.strip() for p in protocols_str.split(",")]

    # 判断使用不同协议(noVNC首选binary)
    selected_protocol = None
    if "binary" in protocols:
        selected_protocol = "binary"
    elif "base64" in protocols:
        selected_protocol = "base64"

    # 使用对应协议接收websocket连接
    logger.info(f"为会话[{session_id}]开启WebSocket连接")
    await websocket.accept(subprotocol=selected_protocol)

    try:
        # 获取对应会话的vnc
        sanbox_vnc_url = await session_service.get_vnc_url(session_id)
        logger.info(f"连接WebSocket VNC: {sanbox_vnc_url}")

        # 创建上下文并连接到vnc
        async with websockets.connect(sanbox_vnc_url) as sandbox_ws:
            # 创建两个异步协程来完成数据的双向转发
            async def forward_to_sandbox():
                try:
                    while True:
                        # 接收来自客户端的数据
                        data = await websocket.receive_bytes()
                        await sandbox_ws.send(data)
                except WebSocketDisconnect:
                    logger.info("Web -> VNC 连接终端")
                except Exception as forward_e:
                    logger.error(f"forward_to_sandbox出错: {str(forward_e)}")

            async def forward_from_sandbox():
                try:
                    while True:
                        # 接收来自沙箱的数据并转发
                        data = await sandbox_ws.recv()
                        if isinstance(data, bytes):
                            await websocket.send_bytes(data)
                        else:
                            await websocket.send_text(data)
                except ConnectionError:
                    logger.info("VNC -> Web连接关闭")
                except Exception as forward_e:
                    logger.error(f"forward_from_sandbox出错: {str(forward_e)}")

            # 并行运行两个任务
            forward_task1 = asyncio.create_task(forward_to_sandbox())
            forward_task2 = asyncio.create_task(forward_from_sandbox())

            # 等待任务任务结果以为WebSocket连接中断
            done, pending = await asyncio.wait(
                [forward_task1, forward_task2],
                return_when=asyncio.FIRST_COMPLETED,
            )
            logger.info("WebSocket连接关闭")

            # 如果任一任务完成则取消其他任务
            for task in pending:
                task.cancel()
    except ConnectionError as connection_e:
        # 连接沙箱失败，关闭websocket
        logger.error(f"连接沙箱失败: {str(connection_e)}")
        await websocket.close(code=1011, reason=f"连接沙箱失败: {str(connection_e)}")
    except Exception as e:
        # 其他错误记录日志并关闭websocket
        logger.error(f"WebSocket异常: {str(e)}")
        await websocket.close(code=1011, reason=f"WebSocket异常: {str(e)}")
