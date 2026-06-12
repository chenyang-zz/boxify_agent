import logging
from uuid import uuid4
import httpx
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional
from app.application.errors.exceptions import AppException, ServerRequestsError
from app.domain.models.app_config import A2AConfig
from app.domain.models.tool_result import ToolResult
from app.domain.services.tools.base import BaseTool, tool

"""
A2A客户端管理器的开发思路
1.在Agent执行过程中，有可能需要多次调用Remote-Agent，
  但是a2a中的agent-card.json请求时网络io，相对耗时，
  所以需要缓存agent-card的相关信息，只有在初始化a2a客户端的时候才初始化一次，
  更新a2a服务器的时候更新，清除a2a客户端管理器时删除;
2.在前端ui交互中，无论a2a服务器是否启动，都会展示Card信息，
  但是，在执行/规划Agent中，我们只传递启用了a2a服务，所以a2a客户端管理器必须动态接收配置;
3.一个a2a客户端会同时管理多个Agent，但是不同的a2a服务器有可能他们的name是一样的，
  需要考虑传递给Agent信息时的唯一性，会配置多一个唯一的id;
4.由于使用httpx客户端，这个客户端需要创建上下文/释放资源，所以可以使用AsyncExitStack来管理
  异步上下文，避免大量使用with..as的嵌套组合;
5.A2AClientManager的初始化非常耗时，一次请求中只初始化一次;
6.A2A配置来自当前用户的数据库配置，所以在使用的时候，最多需要做多一次校验;
7.A2A客户端管理器只实现两个方法，get_remote_agent_card、call_remote_agent;
8.A2A客户端管理器停止时必须清楚对应资源，涵盖了缓存，异步上下文管理器避免资源泄露;
"""

logger = logging.getLogger(__name__)


class A2AClientManager:
    """A2A客户端管理器"""

    def __init__(self, a2a_config: Optional[A2AConfig] = None) -> None:
        """构造函数，完成A2A客户端的初始化"""
        self._a2a_config = a2a_config  # 配置
        self._exit_stack: AsyncExitStack = AsyncExitStack()  # 上下文管理器
        self._httpx_client: Optional[httpx.AsyncClient] = None  # http客户端
        self._agent_cards: Dict[str, Any] = {}  # agent卡片
        self._initialized: bool = False  # 是否初始化

    @property
    def agent_cards(self) -> Dict[str, Any]:
        """只读属性，返回agent卡片信息"""
        return self._agent_cards

    async def initialize(self) -> None:
        """异步初始化函数，用于初始化所有已配置的a2a服务"""
        # 检测是否已经初始化
        if self._initialized:
            return

        try:
            # 初始化httpx客户端
            self._httpx_client = await self._exit_stack.enter_async_context(
                httpx.AsyncClient(timeout=600)
            )

            if not self._a2a_config:
                return

            # 记录日志并连接所有配置的a2a服务获取卡片信息
            logger.info(f"加载{len(self._a2a_config.a2a_servers)}个A2A服务")
            await self._get_a2a_agent_cards()
            self._initialized = True
            logger.info("A2A客户端加载成功")
        except Exception as e:
            logger.error(f"A2A客户端管理器加载失败: {str(e)}")
            raise ServerRequestsError(f"A2A客户端管理器加载失败: {str(e)}")

    async def _get_a2a_agent_cards(self) -> None:
        """根据配置连接所有a2a服务器获取AgentCard信息"""

        if not self._a2a_config:
            return

        if not self._httpx_client:
            raise AppException(msg="httpx客户端没有完成初始化")

        # 循环遍历所有a2a服务
        for a2a_server_config in self._a2a_config.a2a_servers:
            try:
                # 调用httpx客户端发起请求
                agent_card_response = await self._httpx_client.get(
                    f"{a2a_server_config.base_url}/.well-known/agent-caard.json"
                )
                agent_card_response.raise_for_status()
                agent_card = agent_card_response.json()

                # 存储到agent_cards
                agent_card["enabled"] = a2a_server_config.enabled
                self._agent_cards[a2a_server_config.id] = agent_card
            except Exception as e:
                logger.warning(f"加载A2A服务[{a2a_server_config.id}]失败: {str(e)}")
                continue

    async def invoke(self, agetn_id: str, query: str) -> ToolResult:
        """根据传递的智能体id和query调用Remote-Agent"""
        # 判断传递的agent_id是否存在
        if agetn_id not in self._agent_cards:
            return ToolResult(success=False, message="该远程Agent不存在")

        # Agent存在，取出端点信息
        agent_card = self._agent_cards.get(agetn_id, {})
        url = agent_card.get("url", "")

        # 判断端点是否存在
        if url == "":
            return ToolResult(success=False, message="该远程Agent调用端点不存在")

        if not self._httpx_client:
            raise AppException(msg="httpx客户端没有完成初始化")

        try:
            # 使用httpx客户端发起post请求并传递数据
            agent_response = await self._httpx_client.post(
                url,
                json={
                    "id": str(uuid4()),
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": str(uuid4()),
                            "role": "user",
                            "parts": [
                                {"kind": "text", "text": query},
                            ],
                        }
                    },
                },
            )
            agent_response.raise_for_status()
            result = agent_response.json()

            return ToolResult(
                success=True,
                message="调用远程Agent成功",
                data=result,
            )
        except Exception as e:
            logger.error(f"调用远程Agent[{agetn_id}]出错: {str(e)}")
            return ToolResult(
                success=False,
                message=f"调用远程Agent[{agetn_id}]出错: {str(e)}",
            )

    async def cleanup(self) -> None:
        """当退出A2A客户端管理器是，清除对应资源"""
        try:
            await self._exit_stack.aclose()
            self._agent_cards.clear()
            self._initialized = False
            logger.info("清理A2A客户端管理器成功")
        except Exception as e:
            logger.error(f"清理A2A客户端管理器失败: {str(e)}")


class A2ATool(BaseTool):
    """A2A工具包"""

    name: str = "a2a"

    def __init__(self) -> None:
        """构造函数，完成工具包的初始化"""
        super().__init__()
        self._initialized: bool = False
        self.manager: Optional[A2AClientManager] = None

    async def initialize(self, a2a_config: Optional[A2AConfig] = None) -> None:
        """初始化A2A工具包"""
        # 判断是否已初始化
        if not self._initialized:
            # 初始化A2A客户端管理器
            self.manager = A2AClientManager(a2a_config=a2a_config)
            await self.manager.initialize()
            self._initialized = True

    @tool(
        name="get_remote_agent_cards",
        description="获取可远程调用的Agent卡片信息，包含Agent id、名称、描述、技能、请求端点等。",
        parameters={},
        required=[],
    )
    async def get_remote_agent_cards(self) -> ToolResult:
        """获取远程Agent卡片信息列表"""

        if not self.manager:
            raise AppException(msg="A2A客户端管理器未完成初始化")

        # 重组结构，将id填充到agent_card中
        agent_cards = []
        for id, agent_card in self.manager.agent_cards.items():
            agent_cards.append(
                {
                    "id": id,
                    **agent_card,
                }
            )

        # 组装ToolResult响应
        return ToolResult(
            success=True,
            message="获取Agent卡片信息列表成功",
            data=agent_cards,
        )

    @tool(
        name="call_remote_agent",
        description="根据传递的id和query(分配给远程Agent完成的任务query)调用远程Agent完成对应需求",
        parameters={
            "id": {
                "type": "string",
                "description": "需要调用远程agent的id，格式参考get_remote_agent_cards()返回的数据结构",
            },
            "query": {
                "type": "string",
                "description": "需要分配给远程Agent实现的任务/需求query",
            },
        },
        required=["id", "query"],
    )
    async def call_remote_agent(self, id: str, query: str) -> ToolResult:
        """调用远程Agent并完成对应需求"""

        if not self.manager:
            raise AppException(msg="A2A客户端管理器未完成初始化")

        return await self.manager.invoke(agetn_id=id, query=query)
