import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, cast

import httpx
from mcp import ClientSession, StdioServerParameters, Tool, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

from app.application.errors.exceptions import NotFoundError
from app.domain.models.app_config import MCPConfig, MCPServerConfig, MCPTransport
from app.domain.models.tool_result import ToolResult
from app.domain.services.tools.base import BaseTool

"""
MCP客户端管理器开发思路:
1.在Agent执行过程中，有可能需要调用多次工具，
  但是因为MCP工具的每次获取都需要调用客户端的list_tools()方法，
  非常耗时，所以需要缓存工具的参数信息，只有在初始化的时候调用一次，
  并且在销毁MCP客户端管理器的时候一并清除。
2.在前端UI交互中，无论MCP服务器是否启动，都会显示工具列表信息，
  但在Agent执行的过程中，只会传递已启动的MCP服务，
  所以对于MCP客户端管理器来说，可以根据接收的MCP配置的差异加载不同的服务器，
  而不是仅从配置文件中读取数据
3.MCP客户端管理器会同时管理多个MCP服务器，有可能有stdio、see、streamable_http等传输协议，
  需要根据传输协议的不同来创建客户端会话(ClientSession),同时缓存会话。
4.另外有可能有以下环境变量时存在在整个系统中，在初始化MCP服务时，需要将传递的环境变量与系统的
  环境变量进行合并后传递给MCP服务。
5.使用AsyncExitStack异步上下文管理器来管理上下文，避免使用with多层嵌套。
6.MCPClientManager的初始化非常耗时，需要有机制可以判断避免重复初始化。
7.MCP配置来自当前用户的数据库配置，仍需要在初始化前做二次校验，避免无效配置影响运行。
8.同时缓存ClientSession+Tool-Schema，一个是客户端会话，一个是工具参数声明。
9.MCP客户端管理器在清除/停止使用的时候，必须关闭异步上下文管理器、清除资源(ClientSession、Tool-Schema)、
  初始化标识符等，从而避免资源泄露。
"""

logger = logging.getLogger(__name__)


class MCPClientManager:
    """MCP客户端管理器"""

    def __init__(self, mcp_config: Optional[MCPConfig] = None) -> None:
        """构造函数，完成MCP客户端管理器的初步初始化"""
        self._mcp_config: MCPConfig = mcp_config or MCPConfig()  # mcp配置信息
        self._exit_stack: AsyncExitStack = AsyncExitStack()  # 异步上下文管理器
        self._clients: Dict[str, ClientSession] = {}  # 缓存的客户端会话
        self._tools: Dict[str, List[Tool]] = {}  # 缓存的MCP工具参数声明
        self._initialized: bool = False  # 初始化标识

    @property
    def tools(self) -> Dict[str, List[Tool]]:
        """只读属性，返回缓存的MCP工具声明"""
        return self._tools

    async def initialize(self) -> None:
        """初始化函数，用于连接所有配置的MCP服务器"""

        # 1.检查是否已经初始化成功
        if self._initialized:
            return

        try:
            # 2.记录日志并连接MCP服务器
            logger.info(f"正在加载{len(self._mcp_config.mcpServers)}个MCP服务器")
            await self._connect_mcp_servers()
            logger.info("MCP客户端管理器加载成功")
        except Exception as e:
            # 3.记录错误信息并直接抛出
            logger.error(f"MCP客户端管理器加载失败: {str(e)}")
            raise

    async def _connect_mcp_servers(self) -> None:
        """根据配置连接所有MCP服务器"""
        # 1.循环遍历所有传递进来的MCP服务器，不用理会enabled状态，在外部会进行筛选
        for server_name, server_config in self._mcp_config.mcpServers.items():
            try:
                # 2.根据服务名称和服务配置连接到MCP服务器
                await self._connect_mcp_server(server_name, server_config)
            except Exception as e:
                # 3.记录错误日志并跳过错误的MCP服务器
                logger.error(f"连接到MCP服务器[{server_name}]失败: {str(e)}")
                continue

    async def _connect_mcp_server(
        self, server_name: str, server_config: MCPServerConfig
    ) -> None:
        """根据服务名称和服务配置连接到MCP服务器"""

        try:
            # 1.获取mcp服务的传输协议
            transport = server_config.transport

            # 2.根据传输协议连接到不同的mcp服务器
            if transport == MCPTransport.STDIO:
                await self._connect_stdio_server(server_name, server_config)
            elif transport == MCPTransport.SSE:
                await self._connect_sse_server(server_name, server_config)
            elif transport == MCPTransport.STREAMABLE_HTTP:
                await self._connect_streamable_http_server(server_name, server_config)
            else:
                raise ValueError(
                    f"MCP服务[{server_name}]使用了不支持的MCP传输协议: {transport}"
                )

        except Exception as e:
            # 记录错误信息并直接抛出
            logger.error(f"连接到MCP服务器[{server_name}]失败: {str(e)}")
            raise

    async def _connect_stdio_server(
        self, server_name: str, server_config: MCPServerConfig
    ) -> None:
        """连接到STDIO传输协议的MCP服务器"""

        # 1.获取mcp服务的命令和参数
        command = server_config.command
        args = server_config.args
        env = server_config.env

        # 2.验证命令和参数是否配置
        if not command:
            raise ValueError(f"STDIO服务[{server_name}]未配置命令")

        # 3.创建StdioServerParameters对象
        server_parameters = StdioServerParameters(
            command=command,
            args=cast(list, args),
            env={**os.environ, **(env or {})},
        )

        try:
            # 4.使用异步上下文管理器连接到STDIO服务器
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
            read_stream, write_stream = stdio_transport

            # 5.使用异步上下文管理器创建ClientSession
            session = cast(
                ClientSession,
                cast(
                    object,
                    await self._exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    ),
                ),
            )

            # 6.初始化ClientSession
            await session.initialize()

            # 7.将ClientSession存储到客户端字典中
            self._clients[server_name] = session

            # 8.缓存MCP服务器的工具
            await self._cache_mcp_server_tools(server_name, session)
            logger.info(f"连接stdio-mcp服务成功: {server_name}")
        except Exception as e:
            # 记录错误信息并直接抛出
            logger.error(f"连接stdio-mcp服务器[{server_name}]失败: {str(e)}")
            raise

    async def _connect_sse_server(
        self, server_name: str, server_config: MCPServerConfig
    ) -> None:
        """连接SSE服务器"""

        # 1.验证url是否配置
        url = server_config.url
        if not url:
            raise ValueError(f"SSE服务器[{server_name}]的URL未配置")

        try:
            # 2.使用异步上下文管理器创建SSE客户端
            sse_transport = await self._exit_stack.enter_async_context(
                sse_client(url=url, headers=server_config.headers)
            )
            read_stream, write_stream = sse_transport

            # 3.创建ClientSession
            session = cast(
                ClientSession,
                cast(
                    object,
                    await self._exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    ),
                ),
            )

            # 4.初始化ClientSession
            await session.initialize()

            # 5.将ClientSession存储到客户端字典中
            self._clients[server_name] = session

            # 6.缓存MCP服务器的工具
            await self._cache_mcp_server_tools(server_name, session)
            logger.info(f"连接sse-mcp服务成功: {server_name}")
        except Exception as e:
            # 记录错误信息并直接抛出
            logger.error(f"连接sse-mcp服务器[{server_name}]失败: {str(e)}")
            raise

    async def _connect_streamable_http_server(
        self, server_name: str, server_config: MCPServerConfig
    ) -> None:
        """连接Streamable HTTP服务器"""

        # 1.验证url是否配置
        url = server_config.url
        if not url:
            raise ValueError(f"Streamable HTTP服务器[{server_name}]的URL未配置")

        try:
            # 2.使用异步上下文管理器创建streamable http客户端
            streamable_http_transport = await self._exit_stack.enter_async_context(
                streamable_http_client(
                    url=url,
                    http_client=httpx.AsyncClient(headers=server_config.headers),
                )
            )
            read_stream, write_stream, _ = streamable_http_transport

            # 3.创建ClientSession
            session = cast(
                ClientSession,
                cast(
                    object,
                    await self._exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    ),
                ),
            )

            # 4.初始化ClientSession
            await session.initialize()

            # 5.将ClientSession存储到客户端字典中
            self._clients[server_name] = session

            # 6.缓存MCP服务器的工具
            await self._cache_mcp_server_tools(server_name, session)
            logger.info(f"连接streamable-http-mcp服务成功: {server_name}")
        except Exception as e:
            # 记录错误信息并直接抛出
            logger.error(f"连接streamable-http-mcp服务器[{server_name}]失败: {str(e)}")
            raise

    async def _cache_mcp_server_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """缓存MCP服务器的工具"""

        try:
            tools_response = await session.list_tools()
            tools = tools_response.tools if tools_response else []
            self._tools[server_name] = tools
            logger.info(f"MCP服务器[{server_name}]提供了{len(tools)}个工具")
        except Exception as e:
            # 记录错误信息并清空工具缓存
            logger.error(f"获取MCP服务器[{server_name}]的工具失败: {str(e)}")
            self._tools[server_name] = []

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """获取所有MCP工具的OpenAI工具调用格式"""

        # 1.定义一个变量存储所有结果
        all_tools = []

        # 2.遍历所有缓存的工具
        for server_name, tools in self._tools.items():
            # 3.循环取出每个MCP服务的工具列表
            for tool in tools:
                # 4.根据服务器名称生成工具名称
                if server_name.startswith("mcp_"):
                    tool_name = f"{server_name}_{tool.name}"
                else:
                    tool_name = f"mcp_{server_name}_{tool.name}"

                # 5.将工具信息转换为OpenAI工具调用格式
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"[{server_name}] {tool.description or tool.name}",
                        "parameters": tool.inputSchema,
                    },
                }

                all_tools.append(tool_schema)

        return all_tools

    async def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """调用MCP工具"""

        try:
            # 1.解析工具名称获取原始服务器名称和工具名称
            origin_server_name = None
            origin_tool_name = None

            # 2.循环遍历MCP服务配置，解析工具名称获取原始服务器名称和工具名称
            for server_name in self._mcp_config.mcpServers.keys():
                # 3.构建预期的工具名称前缀，用于匹配工具名称
                expected_prefix = (
                    server_name
                    if server_name.startswith("mcp_")
                    else f"mcp_{server_name}"
                )
                # 4.如果工具名称以预期前缀开头，则解析出原始服务器名称和工具名称
                if tool_name.startswith(f"{expected_prefix}_"):
                    # 5.解析出原始服务器名称和工具名称
                    origin_server_name = server_name
                    origin_tool_name = tool_name[len(expected_prefix) + 1 :]
                    break

            # 6.如果未找到对应的MCP服务器，则抛出未找到错误
            if not origin_server_name or not origin_tool_name:
                raise NotFoundError(f"未找到工具[{tool_name}]对应的MCP服务器")

            # 7.获取对应的MCP客户端会话，如果不存在则返回失败结果
            session = self._clients.get(origin_server_name)
            if not session:
                return ToolResult(
                    success=False, message=f"未找到MCP服务器[{origin_server_name}]"
                )

            # 8.调用MCP工具并获取结果
            result = await session.call_tool(origin_tool_name, arguments)

            # 9.如果结果存在，则提取文本内容并返回成功结果
            if result:
                # 10.提取文本内容
                content = []
                if hasattr(result, "content") and result.content:
                    for item in result.content:
                        if hasattr(result, "text"):
                            result = cast(TextContent, item)
                            content.append(result.text)
                        else:
                            content.append(str(item))

                # 11.返回成功结果
                return ToolResult(
                    success=True, data="\n".join(content) if content else "工具执行成功"
                )
            else:
                return ToolResult(success=True, message="工具执行成功")

        except Exception as e:
            # 记录错误信息并返回失败结果
            logger.error(f"调用MCP工具[{tool_name}]失败: {str(e)}")
            return ToolResult(
                success=False, message=f"调用MCP工具[{tool_name}]失败: {str(e)}"
            )

    async def cleanup(self) -> None:
        """当退出MCP服务时，清除对应资源

        该方法是幂等的，多次调用不会产生副作用。
        注意：必须在初始化MCP的同一个asyncio Task中调用此方法，
        否则anyio会因cancel scope上下文不匹配而抛出RuntimeError。
        """

        # 幂等检查：如果未初始化则跳过清理
        if not self._initialized:
            return
        try:
            await self._exit_stack.aclose()
            logger.info("清除MCP客户端管理器成功")
        except RuntimeError as e:
            # 防御性处理：anyio.create_task_group() 在不同任务中退出的已知问题
            if "Attempted to exit cancel scope in a different task" in str(e):
                logger.warning(
                    f"清理MCP客户端管理器时遇到任务上下文切换警告（可忽略）: {str(e)}"
                )
            else:
                logger.error(f"清理MCP客户端管理器失败: {str(e)}")
        except Exception as e:
            logger.error(f"清理MCP客户端管理器失败: {str(e)}")
        finally:
            # 无论aclose()是否成功，都必须清除缓存并重置状态
            self._clients.clear()
            self._tools.clear()
            self._initialized = False


class MCPTool(BaseTool):
    """MCP工具包，包含所有已配置、已启动的MCP工具"""

    name: str = "mcp"

    def __init__(self) -> None:
        """构造函数，完成MCP工具包的初始化"""
        super().__init__()

        self._initialized = False
        self._tools = []
        self._manager: Optional[MCPClientManager] = None

    async def initialize(self, mcp_config: Optional[MCPConfig] = None) -> None:
        """初始化MCP工具包"""

        # 1. 判断是否初始化
        if not self._initialized:
            # 2. 创建MCPClientManager并初始化
            self._manager = MCPClientManager(mcp_config=mcp_config)
            manager: MCPClientManager = cast(MCPClientManager, self._manager)
            await manager.initialize()

            # 3. 获取所有工具
            self._tools = manager.get_all_tools()
            self._initialized = True

    def get_tools(self) -> List[Dict[str, Any]]:
        """同步获取工具包下的所有工具列表"""
        return self._tools

    def has_tool(self, tool_name: str) -> bool:
        """判断工具包下是否存在指定的工具"""
        for tool in self._tools:
            if tool["function"]["name"] == tool_name:
                return True
        return False

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        """调用指定工具"""

        if not self._initialized or self._manager:
            raise RuntimeError("MCP工具包未初始化")

        manager: MCPClientManager = cast(MCPClientManager, self._manager)
        return await manager.invoke(tool_name, kwargs)

    async def cleanup(self) -> None:
        """清理MCP工具包资源"""
        if self._manager:
            await self._manager.cleanup()
            self._manager = None
            self._initialized = False
