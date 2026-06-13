from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMConfig(BaseModel):
    """语言模型配置"""

    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model_name: str = "deepseek-v4-pro"
    temperature: float = Field(default=0.7)  # 温度
    max_tokens: int = Field(default=200 * 1024, ge=0)  # 最大输出token数


class NotebookEmbeddingConfig(BaseModel):
    """Notebook知识库Embedding模型配置"""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "text-embedding-3-small"


class NotebookConfig(BaseModel):
    """Notebook知识库配置"""

    embedding_config: NotebookEmbeddingConfig = Field(
        default_factory=NotebookEmbeddingConfig
    )


class AgentConfig(BaseModel):
    """Agent通用配置"""

    max_iterations: int = Field(default=100, gt=0, lt=1000)  # 最大迭代次数
    max_retries: int = Field(default=3, gt=1, lt=10)  # 最大重试次数
    max_search_results: int = Field(default=10, gt=1, lt=30)  # 最大搜索结果数


class MCPTransport(str, Enum):
    """MCP传输类型枚举"""

    STDIO = "stdio"  # 本地输入输出
    SSE = "sse"  # 流式事件类型
    STREAMABLE_HTTP = "streamable-http"  # 可流式的http


class MCPServerConfig(BaseModel):
    """MCP单条服务配置"""

    # 通用字段配置
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP  # 传输协议
    enabled: bool = True  # 是否启用
    description: Optional[str] = None  # MCP服务描述
    env: Optional[Dict[str, Any]] = None  # 环境变量

    # stdio配置
    command: Optional[str] = None  # 启动命令
    args: Optional[List[str]] = None  # 命令参数

    # streamable_http和sse配置
    url: Optional[str] = None  # MCP服务URL地址
    headers: Optional[Dict[str, Any]] = None  # headers请求头

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_mcp_service_config(self):
        """校验mcp_service_config相关信息，包含url+command"""

        # 1.判断transport是否为see/streamable_http
        if self.transport in [MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP]:
            # 2.判断url是否为空
            if not self.url:
                raise ValueError("在sse或streamable_http传输协议中必须传递url")

        # 3.判断transport是否为stdio模式
        if self.transport == MCPTransport.STDIO:
            # 4.判断command是否为空
            if not self.command:
                raise ValueError("在stdio模式中必须传递command")

        return self


class MCPConfig(BaseModel):
    """应用MCP配置"""

    mcpServers: Dict[str, MCPServerConfig] = Field(default_factory=dict)  # MCP服务

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class A2AServerConfig(BaseModel):
    """A2A服务配置"""

    id: str = Field(default_factory=lambda: str(uuid4()))  # 唯一标识
    base_url: str  # 基础服务URL
    enabled: bool = True  # 服务是否开启


class A2AConfig(BaseModel):
    """A2A配置"""

    a2a_servers: List[A2AServerConfig] = Field(default_factory=list)


class AppConfig(BaseModel):
    """应用配置信息，包含Agent配置、A2A网络、LLM供应商、MCP服务配置等"""

    llm_config: LLMConfig  # 语言模型配置
    agent_config: AgentConfig  # Agent通用配置
    mcp_config: MCPConfig  # MCP配置
    a2a_config: A2AConfig  # A2A服务配置
    notebook_config: NotebookConfig = Field(default_factory=NotebookConfig)

    # Pydantic配置，允许传递额外的参数初始化
    model_config = ConfigDict(extra="allow")


def create_default_app_config() -> AppConfig:
    """创建系统默认应用配置"""
    return AppConfig(
        llm_config=LLMConfig(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        notebook_config=NotebookConfig(),
    )
