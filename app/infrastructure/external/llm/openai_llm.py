import logging
from typing import Any, AsyncIterator, Dict, List, cast

from openai import AsyncOpenAI, Omit
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import ResponseFormat

from app.application.errors.exceptions import ServerRequestsError
from app.domain.external.llm import LLM, LLMMessage, LLMTextStream
from app.domain.models.app_config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """基于OpenAI SDK/兼容OpenAI格式的LLM调用类"""

    def __init__(self, llm_config: LLMConfig, **kwargs) -> None:
        """构造函数，完成异步OpenAI客户端的创建和参数初始化"""
        # 1.初始化异步客户端
        self._client = AsyncOpenAI(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key or None,
            **kwargs,
        )

        # 2.完成其他参数的存储
        self._model_name = llm_config.model_name
        self._temperature = llm_config.temperature
        self._max_tokens = llm_config.max_tokens
        self._timeout = 3600

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        response_format: Dict[str, Any] | None = None,
        tool_choice: str | None = None,
    ) -> LLMMessage:
        """使用异步OpenAI客户端发起完整响应请求。"""

        openai_messages = cast(list[ChatCompletionMessageParam], messages)
        openai_tools = cast(list[ChatCompletionFunctionToolParam] | Omit, tools)
        openai_response_format = cast(ResponseFormat | Omit, response_format)
        openai_tool_choice = cast(
            ChatCompletionToolChoiceOptionParam | Omit, tool_choice
        )

        try:
            # 1.检查是否传递了工具列表
            if tools:
                logger.info(
                    f"调用OpenAI客户端向LLM发起请求并携带工具信息: {self._model_name}"
                )
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=openai_messages,
                    response_format=openai_response_format,
                    tools=openai_tools,
                    tool_choice=openai_tool_choice,
                    parallel_tool_calls=False,  # 关闭并行工具调用(deepseek没有这个参数)
                    timeout=self._timeout,
                )
            else:
                # 2.未传递工具则删除tools/tools_choice等参数
                logger.info(
                    f"调用OpenAI客户端向LLM发起请求并未携带工具: {self._model_name}"
                )
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=openai_messages,
                    response_format=openai_response_format,
                    timeout=self._timeout,
                )

            # 3.处理响应数据并返回
            logger.info(f"OpenAI客户端返回内容: {response.model_dump()}")
            return response.choices[0].message.model_dump()
        except ServerRequestsError:
            raise
        except Exception as e:
            logger.info(f"调用OpenAI客户端发生错误: {str(e)}")
            raise ServerRequestsError("调用OpenAI客户端向LLM发起请求出错")

    async def stream(self, messages: List[Dict[str, Any]]) -> LLMTextStream:
        """使用异步OpenAI客户端发起纯文本流式响应请求。"""
        openai_messages = cast(list[ChatCompletionMessageParam], messages)
        try:
            logger.info(f"调用OpenAI客户端向LLM发起流式请求: {self._model_name}")
            response_stream = await self._client.chat.completions.create(
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=openai_messages,
                stream=True,
                timeout=self._timeout,
            )
            async for chunk in self._iter_text_chunks(
                cast(AsyncIterator[Any], response_stream)
            ):
                yield chunk
        except ServerRequestsError:
            raise
        except Exception as e:
            logger.info(f"调用OpenAI客户端发生流式响应错误: {str(e)}")
            raise ServerRequestsError("调用OpenAI客户端向LLM发起请求出错") from e

    async def _iter_text_chunks(
        self, response_stream: AsyncIterator[Any]
    ) -> AsyncIterator[str]:
        """从 OpenAI 流式响应中提取纯文本片段。"""
        try:
            async for chunk in response_stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as e:
            logger.info(f"读取OpenAI流式响应发生错误: {str(e)}")
            raise ServerRequestsError("调用OpenAI客户端向LLM发起请求出错") from e


if __name__ == "__main__":
    import asyncio

    async def main():
        llm = OpenAILLM(LLMConfig())
        response = await llm.invoke([{"role": "user", "content": "Hi"}])
        print(response)

    asyncio.run(main())
