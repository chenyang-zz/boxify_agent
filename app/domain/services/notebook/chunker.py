from dataclasses import dataclass

from app.domain.models.knowledge import KnowledgeChunk


@dataclass
class ParentChunk:
    """父子分块结果，父块用于回显上下文，子块用于向量召回。"""

    content: str
    children: list[str]


class DocumentChunker:
    """Notebook文档分块领域服务，统一控制父子 chunk 的窗口策略。"""

    CHUNK_TYPE_PARENT = "parent"
    CHUNK_TYPE_CHILD = "child"

    @classmethod
    def chunk_parent_child(
        cls,
        text: str,
        parent_size: int = 1600,
        child_size: int = 500,
        overlap: int = 80,
    ) -> list[ParentChunk]:
        """按固定窗口生成父块和子块，保留重叠文本提升召回连续性。"""
        normalized = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        if not normalized:
            return []
        parents = cls._window(normalized, parent_size, overlap)
        return [
            ParentChunk(
                content=parent,
                children=cls._window(parent, child_size, overlap),
            )
            for parent in parents
        ]

    @classmethod
    def build_chunk(
        cls,
        user_id: str,
        source_id: str,
        doc_name: str,
        content: str,
        chunk_type: str,
        vector: list[float] | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> KnowledgeChunk:
        """构建领域层知识 chunk，基础设施层负责转换为具体索引文档。"""
        return KnowledgeChunk(
            user_id=user_id,
            source_id=source_id,
            doc_name=doc_name,
            chunk_type=chunk_type,
            parent_id=parent_id,
            content=content,
            vector=vector,
            tags=tags or [],
        )

    @staticmethod
    def _window(text: str, size: int, overlap: int) -> list[str]:
        """按字符窗口切分文本；中文场景下避免依赖空格分词。"""
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        step = max(size - overlap, 1)
        while start < len(text):
            chunk = text[start : start + size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks
