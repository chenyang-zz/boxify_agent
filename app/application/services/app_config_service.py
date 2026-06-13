from typing import List
from uuid import uuid4

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
    NotebookEmbeddingConfig,
)
from app.domain.repositories.vow import IUnitOfWork
from app.domain.services.tools.a2a import A2AClientManager
from app.domain.services.tools.mcp import MCPClientManager
from app.interfaces.schemas.app_config import ListA2AServerItem, ListMCPServerItem


class AppConfigService:
    """应用配置服务"""

    def __init__(self, uow_factory, user_id: str):
        """构造函数，完成应用配置服务的初始化"""
        self._uow_factory = uow_factory
        self._user_id = user_id

    async def _load_app_config(self, uow: IUnitOfWork) -> AppConfig:
        """加载获取所有的应用配置"""
        return await uow.app_config.get_or_create_default(self._user_id)

    async def get_app_config(self) -> AppConfig:
        """获取当前用户完整应用配置"""
        async with self._uow_factory() as uow:
            return await self._load_app_config(uow)

    async def get_llm_config(self) -> LLMConfig:
        """获取LLM提供商配置"""
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)
            return app_config.llm_config

    async def update_llm_config(self, llm_config: LLMConfig) -> LLMConfig:
        """根据传递的llm_config更新语言模型提供商配置"""
        async with self._uow_factory() as uow:
            # 1.获取应用配置
            app_config = await self._load_app_config(uow)

            # 2.判断api_key是否为空
            if not llm_config.api_key.strip():
                llm_config.api_key = app_config.llm_config.api_key

            # 3.调用函数更新app_config
            app_config.llm_config = llm_config
            await uow.app_config.save(self._user_id, app_config)

            return app_config.llm_config

    async def get_notebook_embedding_config(self) -> NotebookEmbeddingConfig:
        """获取Notebook知识库Embedding配置"""
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)
            return app_config.notebook_config.embedding_config

    async def update_notebook_embedding_config(
        self, embedding_config: NotebookEmbeddingConfig
    ) -> NotebookEmbeddingConfig:
        """更新Notebook知识库Embedding配置"""
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)
            if not embedding_config.api_key.strip():
                embedding_config.api_key = (
                    app_config.notebook_config.embedding_config.api_key
                )
            app_config.notebook_config.embedding_config = embedding_config
            await uow.app_config.save(self._user_id, app_config)
            return app_config.notebook_config.embedding_config

    async def get_agent_config(self) -> AgentConfig:
        """获取Agent通用配置"""
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)
            return app_config.agent_config

    async def update_agent_config(self, agent_config: AgentConfig) -> AgentConfig:
        """根据传递的agent_config更新Agent通用配置"""
        async with self._uow_factory() as uow:
            # 1.获取应用配置
            app_config = await self._load_app_config(uow)

            # 3.调用函数更新app_config
            app_config.agent_config = agent_config
            await uow.app_config.save(self._user_id, app_config)

            return app_config.agent_config

    async def get_mcp_servers(self) -> List[ListMCPServerItem]:
        """获取MCP服务器列表"""

        # 1.获取应用配置
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)

        # 2.创建MCP客户端管理器
        mcp_servers = []
        mcp_server_manager = MCPClientManager(mcp_config=app_config.mcp_config)

        try:
            # 3.初始化MCP客户端管理器
            await mcp_server_manager.initialize()

            # 4.获取MCP工具声明
            tools = mcp_server_manager.tools

            # 5.构建MCP服务器列表
            for server_name, server_config in app_config.mcp_config.mcpServers.items():
                mcp_servers.append(
                    ListMCPServerItem(
                        server_name=server_name,
                        enabled=server_config.enabled,
                        transport=server_config.transport,
                        tools=[tool.name for tool in tools.get(server_name, [])],
                    )
                )
        finally:
            # 6.清理MCP客户端管理器
            await mcp_server_manager.cleanup()

        return mcp_servers

    async def update_and_create_mcp_servers(self, mcp_config: MCPConfig) -> MCPConfig:
        """根据传递的mcp_config更新MCP服务配置"""

        async with self._uow_factory() as uow:
            # 1.获取应用配置
            app_config = await self._load_app_config(uow)

            # 2.更新mcp_config
            app_config.mcp_config.mcpServers.update(mcp_config.mcpServers)

            # 3.调用数据仓库完成存储和更新
            await uow.app_config.save(self._user_id, app_config)

            return app_config.mcp_config

    async def delete_mcp_server(self, server_name: str) -> MCPConfig:
        """根据MCP服务名称删除对应的MCP配置"""

        async with self._uow_factory() as uow:
            # 1.获取应用配置
            app_config = await self._load_app_config(uow)

            # 2.查询对应的服务是否存在
            if server_name not in app_config.mcp_config.mcpServers:
                raise NotFoundError(f"未找到MCP服务: {server_name}")

            # 3.存在则删除服务
            del app_config.mcp_config.mcpServers[server_name]

            await uow.app_config.save(self._user_id, app_config)

            return app_config.mcp_config

    async def set_mcp_server_enabled(
        self, server_name: str, enabled: bool
    ) -> MCPConfig:
        """更新MCP服务启用状态"""

        async with self._uow_factory() as uow:
            # 1.获取应用配置
            app_config = await self._load_app_config(uow)

            # 2.查询对应的服务是否存在
            if server_name not in app_config.mcp_config.mcpServers:
                raise NotFoundError(f"未找到MCP服务: {server_name}")

            # 3.更新服务启用状态
            app_config.mcp_config.mcpServers[server_name].enabled = enabled

            await uow.app_config.save(self._user_id, app_config)

            return app_config.mcp_config

    async def get_a2a_servers(self) -> List[ListA2AServerItem]:
        """获取A2A服务列表"""
        # 获取当前的应用配置
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)

        # 构建a2a客户端管理器，对配置信息不过滤
        a2a_servers = []
        a2a_client_manager = A2AClientManager(app_config.a2a_config)

        try:
            # 初始化a2a客户端管理器
            await a2a_client_manager.initialize()

            # 获取Agent卡片信息
            agent_cards = a2a_client_manager.agent_cards

            # 组装响应结构
            for id, agent_card in agent_cards.items():
                a2a_servers.append(
                    ListA2AServerItem(
                        id=id,
                        name=agent_card.get("name", ""),
                        description=agent_card.get("description", ""),
                        input_modes=agent_card.get("defaultInputModes", []),
                        output_modes=agent_card.get("defaultOutputModes", []),
                        streaming=agent_card.get("capabilities", {}).get(
                            "streaming", False
                        ),
                        push_notifications=agent_card.get("capabilities", {}).get(
                            "push_notifications", False
                        ),
                        enabled=agent_card.get("enabled", False),
                    )
                )
        finally:
            # 清除客户端管理器资源
            await a2a_client_manager.cleanup()

        return a2a_servers

    async def create_a2a_server(self, base_url: str) -> A2AConfig:
        """根据传递的配置新增a2a服务器"""
        # 获取当前的应用配置
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)

            # 往数据中新增a2a服务(在新增之前可以检测下当前Agent是否存在)
            for a2a_server_config in app_config.a2a_config.a2a_servers:
                if a2a_server_config.base_url == base_url:
                    raise BadRequestError(msg="当前a2a服务器已经存在")

            a2a_server_config = A2AServerConfig(
                id=str(uuid4()),
                base_url=base_url,
                enabled=True,
            )
            app_config.a2a_config.a2a_servers.append(a2a_server_config)

            # 调用数据仓库更新
            await uow.app_config.save(self._user_id, app_config)
            return app_config.a2a_config

    async def set_a2a_server_enabled(self, a2a_id: str, enabled: bool) -> A2AConfig:
        """根据传递的id和enabled更新服务启用状态"""
        # 获取当前的应用配置
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)

            # 计算需要更新位置的索引并判断是否存在
            idx = None
            for item_idx, item in enumerate(app_config.a2a_config.a2a_servers):
                if item.id == a2a_id:
                    idx = item_idx
                    break

            if idx is None:
                raise NotFoundError(msg=f"该A2A服务[{a2a_id}]不存在，请核实后重试")

            # 存在则更新数据
            app_config.a2a_config.a2a_servers[idx].enabled = enabled
            await uow.app_config.save(self._user_id, app_config)

            return app_config.a2a_config

    async def delete_a2a_server(self, a2a_id: str) -> A2AConfig:
        """根据传递的id删除指定的a2a服务"""
        # 获取当前的应用配置
        async with self._uow_factory() as uow:
            app_config = await self._load_app_config(uow)

            # 计算需要删除位置的索引并判断是否存在
            idx = None
            for item_idx, item in enumerate(app_config.a2a_config.a2a_servers):
                if item.id == a2a_id:
                    idx = item_idx
                    break

            if idx is None:
                raise NotFoundError(msg=f"该A2A服务[{a2a_id}]不存在，请核实后重试")

            # 删除a2a服务器
            del app_config.a2a_config.a2a_servers[idx]
            await uow.app_config.save(self._user_id, app_config)

            return app_config.a2a_config
