from typing import Protocol


class EmbeddingModel(Protocol):
    """Embedding模型接口协议"""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表，返回顺序必须与输入顺序一致。"""
        ...

    async def embed_one(self, text: str) -> list[float]:
        """将单条文本转换为向量。"""
        ...

    @property
    def model_name(self) -> str:
        """只读属性，返回 Embedding 模型名字。"""
        ...
