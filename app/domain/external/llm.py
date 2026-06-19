from typing import Any, AsyncIterator, Dict, List, Protocol

LLMMessage = Dict[str, Any]
LLMTextStream = AsyncIterator[str]


class LLM(Protocol):
    """用于Agent应用与LLM进行交互的接口协议"""

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        response_format: Dict[str, Any] | None = None,
        tool_choice: str | None = None,
    ) -> LLMMessage:
        """传递消息列表、工具列表、响应格式，工具选择策略调用LLM接口"""
        ...

    def stream(
        self,
        messages: List[Dict[str, Any]],
    ) -> LLMTextStream:
        """以纯文本片段流式调用LLM接口"""
        ...

    @property
    def model_name(self) -> str:
        """只读属性，返回LLM的模型名字"""
        ...

    @property
    def temperature(self) -> float:
        """只读属性，返回LLM的温度"""
        ...

    @property
    def max_tokens(self) -> int:
        """只读属性，返回LLM的最大token数"""
        ...
