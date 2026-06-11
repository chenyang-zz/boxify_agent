import logging
from typing import Dict, Optional

from fastapi import APIRouter, Body, Depends

from app.application.services.app_config_service import AppConfigService
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    LLMConfig,
    MCPConfig,
)
from app.interfaces.schemas.app_config import (
    ListA2AserverResponse,
    ListMCPServerResponse,
)
from app.interfaces.schemas.base import Response
from app.interfaces.service_dependencies import get_app_config_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/app-config", tags=["设置模块"])


@router.get(
    path="/llm",
    response_model=Response[LLMConfig],
    summary="获取LLM配置信息",
    description="包含LLM提供商的base_url、temperature、model_name、max_tokens",
)
async def get_llm_config(
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """获取LLM配置信息"""
    llm_config = await app_config_service.get_llm_config()
    return Response.success(data=llm_config.model_dump(exclude={"api_key"}))


@router.post(
    path="/llm",
    response_model=None,
    summary="更新LLM配置信息",
    description="更新LLM配置信息，当api_key为空的时候表示不更新该字段",
)
async def update_llm_config(
    new_llm_config: LLMConfig,
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """更新LLM配置信息"""
    updated_llm_config = await app_config_service.update_llm_config(new_llm_config)
    return Response.success(
        msg="更新LLM信息配置成功",
        data=updated_llm_config.model_dump(exclude={"api_key"}),
    )


@router.get(
    path="/agent",
    response_model=Response[AgentConfig],
    summary="获取Agent通用配置信息",
    description="包含最大迭代次数、最大重试次数、最大搜索结果数",
)
async def get_agent_config(
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """获取Agent通用配置信息"""
    agent_config = await app_config_service.get_agent_config()
    return Response.success(data=agent_config.model_dump())


@router.post(
    path="/agent",
    response_model=Response[AgentConfig],
    summary="更新Agent通用配置信息",
    description="更新Agent通用配置信息",
)
async def update_agent_config(
    new_agent_config: AgentConfig,
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """更新Agent配置信息"""
    updated_agent_config = await app_config_service.update_agent_config(
        new_agent_config
    )
    return Response.success(
        msg="更新Agent通用信息配置成功",
        data=updated_agent_config.model_dump(),
    )


@router.get(
    path="/mcp-servers",
    response_model=Response[ListMCPServerResponse],
    summary="获取MCP服务器工具列表",
    description="获取当前系统的MCP服务器列表，包含MCP服务名称、工具名称、启用状态等",
)
async def get_mcp_servers(
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """获取当前系统的MCP服务器工具列表"""
    mcp_servers = await app_config_service.get_mcp_servers()
    return Response.success(
        msg="获取mcp服务器列表成功",
        data=ListMCPServerResponse(mcp_servers=mcp_servers).model_dump(),
    )


@router.post(
    path="/mcp-servers",
    response_model=Response[Optional[MCPConfig]],
    summary="新增MCP服务配置，支持传递一个或者多个配置",
    description="传递MCP配置信息为系统新增MCP工具",
)
async def add_mcp_servers(
    mcp_config: MCPConfig,
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """根据传递的mcp配置信息创建mcp服务"""
    updated_mcp_config = await app_config_service.update_and_create_mcp_servers(
        mcp_config
    )
    return Response.success(
        msg="新增MCP配置服务成功",
        data=updated_mcp_config.model_dump(),
    )


@router.post(
    path="/mcp-servers/{server_name}/delete",
    response_model=Response[Optional[MCPConfig]],
    summary="删除MCP服务配置",
    description="根据MCP服务名称删除对应的MCP配置",
)
async def delete_mcp_server(
    server_name: str,
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """根据MCP服务名称删除对应的MCP配置"""
    deleted_mcp_config = await app_config_service.delete_mcp_server(server_name)
    return Response.success(
        msg="删除MCP配置服务成功",
        data=deleted_mcp_config.model_dump(),
    )


@router.post(
    path="/mcp-servers/{server_name}/enabled",
    response_model=Response[Optional[MCPConfig]],
    summary="更新MCP服务的启动状态",
    description="根据传递的server_name和enabled更新指定MCP服务的启用状态",
)
async def set_mcp_server_enabled(
    server_name: str,
    enabled: bool = Body(...),
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """根据server_name和enabled更新对应的MCP服务的启动状态"""
    enabled_mcp_config = await app_config_service.set_mcp_server_enabled(
        server_name, enabled
    )
    return Response.success(
        msg="更新MCP服务启动状态成功", data=enabled_mcp_config.model_dump()
    )


@router.get(
    path="/a2a-servers",
    summary="获取a2a服务器列表",
    description="获取Boxify项目中的所有已配置的a2a服务列表",
    response_model=Response[ListMCPServerResponse],
)
async def get_a2a_servers(
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """获取a2a服务列表"""
    a2a_servers = await app_config_service.get_a2a_servers()
    return Response.success(
        msg="获取a2a服务列表成功",
        data=ListA2AserverResponse(
            a2a_servers=a2a_servers,
        ),
    )


@router.post(
    path="/a2a-servers",
    summary="新增a2a服务器",
    description="为Boxify项目新增a2a服务器",
    response_model=Response[A2AConfig],
)
async def create_a2a_server(
    base_url: str = Body(..., embed=True),
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """新增a2a服务器"""
    created_server = await app_config_service.create_a2a_server(base_url)
    return Response.success(
        msg="新增A2A服务配置成功",
        data=created_server,
    )


@router.post(
    path="/a2a-servers/{a2a_id}/delete",
    summary="删除a2a服务器",
    description="根据A2A服务id表示删除指定的A2A服务",
    response_model=Response[Optional[Dict]],
)
async def delete_a2a_server(
    a2a_id: str,
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    await app_config_service.delete_a2a_server(a2a_id)
    return Response.success(msg="删除a2a服务器成功")


@router.post(
    path="/a2a-servers/{a2a_id}/enabled",
    summary="更新A2A服务器的启用状态",
    description="启用or禁用A2A服务的状态",
    response_model=Response[Optional[Dict]],
)
async def set_a2a_server_enabled(
    a2a_id: str,
    enabled: bool = Body(..., embed=True),
    app_config_service: AppConfigService = Depends(get_app_config_service),
):
    """更新A2A服务的启用状态"""
    await app_config_service.set_a2a_server_enabled(a2a_id, enabled)
    return Response.success(msg="更新a2a服务器启用状态成功")
