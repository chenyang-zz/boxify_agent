from openai import AsyncOpenAI

from app.domain.external.embedding import EmbeddingModel
from app.domain.models.app_config import NotebookEmbeddingConfig


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI 兼容 Embedding 客户端，使用用户独立配置初始化。"""

    def __init__(self, config: NotebookEmbeddingConfig) -> None:
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key or None,
        )
        self._model_name = config.model_name

    @property
    def model_name(self) -> str:
        """返回当前客户端使用的模型名，便于日志和测试断言。"""
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成向量，保持 OpenAI SDK 返回顺序。"""
        response = await self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str) -> list[float]:
        """生成单条文本向量，复用批量接口以保持行为一致。"""
        return (await self.embed([text]))[0]
