from typing import Callable

from elasticsearch.helpers import async_bulk

from app.application.errors.exceptions import BadRequestError
from app.domain.external.embedding import EmbeddingModel
from app.domain.external.knowledge_search import KnowledgeSearch
from app.domain.models.knowledge import KnowledgeChunk, KnowledgeSearchHit
from app.domain.repositories.vow import IUnitOfWork
from app.domain.services.notebook.chunker import DocumentChunker
from app.infrastructure.storage.elasticsearch import get_elasticsearch
from core.config import get_settings

_VECTOR_WEIGHT = 0.6
_BM25_WEIGHT = 0.4
CHUNKS_INDEX = "boxify_notebook_chunks"
_IK_PLUGIN_NAME = "analysis-ik"
_IK_PLUGIN_INSTALL_HINT = (
    "elasticsearch-plugin install "
    "https://get.infini.cloud/elasticsearch/analysis-ik/8.17.0"
)


class ESKnowledgeSearch(KnowledgeSearch):
    """基于 Elasticsearch 的知识库检索服务。"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        embedding: EmbeddingModel,
    ) -> None:
        self._uow_factory = uow_factory
        self._embedding = embedding
        self._settings = get_settings()
        self._client = get_elasticsearch().client

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[KnowledgeSearchHit]:
        """执行用户隔离的混合检索，并返回面向接口的命中结构。"""
        return await self._hybrid_search(
            user_id,
            query=query,
            top_k=top_k,
            tags=tags,
        )

    async def _hybrid_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        recall_size: int = 20,
        tags: list[str] | None = None,
    ) -> list[KnowledgeSearchHit]:
        """融合向量召回和 BM25 召回，兼顾语义相关性与关键词精确匹配。"""
        base_filter: list[dict] = [
            {"term": {"user_id": user_id}},
            {"term": {"source_type": "document"}},
            {"term": {"chunk_type": DocumentChunker.CHUNK_TYPE_CHILD}},
        ]
        if tags:
            base_filter.append({"terms": {"tags": tags}})

        query_vector = await self._embedding.embed_one(query)
        knn_resp = await self._client.search(
            index=CHUNKS_INDEX,
            body={
                "size": recall_size,
                "knn": {
                    "field": "vector",
                    "query_vector": query_vector,
                    "k": recall_size,
                    "num_candidates": recall_size * 5,
                    "filter": {"bool": {"filter": base_filter}},
                },
            },
        )
        bm25_resp = await self._client.search(
            index=CHUNKS_INDEX,
            body={
                "size": recall_size,
                "query": {
                    "bool": {
                        "must": [{"match": {"content": query}}],
                        "filter": base_filter,
                    }
                },
            },
        )

        hits: dict[str, dict] = {}
        vector_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        for hit in knn_resp["hits"]["hits"]:
            hits[hit["_id"]] = hit["_source"]
            vector_scores[hit["_id"]] = hit["_score"]
        for hit in bm25_resp["hits"]["hits"]:
            hits[hit["_id"]] = hit["_source"]
            bm25_scores[hit["_id"]] = hit["_score"]

        # 两种检索分数尺度不同，先归一化再按固定权重融合。
        vector_normalized = self._normalize(vector_scores)
        bm25_normalized = self._normalize(bm25_scores)
        fused = {
            chunk_id: _VECTOR_WEIGHT * vector_normalized.get(chunk_id, 0)
            + _BM25_WEIGHT * bm25_normalized.get(chunk_id, 0)
            for chunk_id in hits
        }

        ranked_ids = [
            chunk_id
            for chunk_id, _ in sorted(
                fused.items(), key=lambda item: item[1], reverse=True
            )
        ][:top_k]
        results = []
        for chunk_id in ranked_ids:
            source = hits[chunk_id]
            content = await self._resolve_parent_content(user_id, source)
            results.append(
                KnowledgeSearchHit(
                    chunk_id=chunk_id,
                    content=content,
                    doc_name=source.get("doc_name"),
                    source_id=source.get("source_id"),
                    source_type=source.get("source_type"),
                    score=round(fused.get(chunk_id, 0), 4),
                )
            )
        return results

    async def delete_by_source(self, user_id: str, document_id: str) -> None:
        """按用户和文档删除旧 chunk，解析重试和删除文档都会复用。"""
        await self._client.delete_by_query(
            index=CHUNKS_INDEX,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": user_id}},
                            {"term": {"source_id": document_id}},
                        ]
                    }
                }
            },
            conflicts="proceed",
            refresh=True,
        )

    async def ensure_index(self) -> None:
        """确保知识库 chunk 索引存在。"""
        await self._ensure_required_plugins()
        if await self._client.indices.exists(index=CHUNKS_INDEX):
            return
        await self._client.indices.create(
            index=CHUNKS_INDEX,
            body=self._build_chunks_index_body(self._settings.notebook_embedding_dims),
        )

    async def _ensure_required_plugins(self) -> None:
        """确认所有 ES 节点都安装 IK 分词插件，避免创建索引时才暴露模糊错误。"""
        nodes_info = await self._client.nodes.info(metric="plugins")
        nodes = nodes_info.get("nodes", {})
        if not nodes:
            raise RuntimeError(
                "无法获取Elasticsearch节点插件信息，无法确认是否安装"
                f"{_IK_PLUGIN_NAME}插件。请检查ES节点状态。"
            )

        missing_nodes = []
        for node_id, node_info in nodes.items():
            plugin_names = {
                plugin.get("name") for plugin in node_info.get("plugins", [])
            }
            if _IK_PLUGIN_NAME not in plugin_names:
                missing_nodes.append(node_info.get("name") or node_id)

        if missing_nodes:
            raise RuntimeError(
                f"Notebook Elasticsearch缺少{_IK_PLUGIN_NAME}插件，"
                f"缺失节点: {', '.join(missing_nodes)}。"
                f"请在每个ES节点执行: {_IK_PLUGIN_INSTALL_HINT}，"
                "并重启Elasticsearch。"
            )

    async def save_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """批量保存领域层 chunk 到 Elasticsearch。"""
        if not chunks:
            return

        actions = [
            {
                "_op_type": "index",
                "_index": CHUNKS_INDEX,
                "_id": chunk.id,
                "_source": self._chunk_to_source(chunk),
            }
            for chunk in chunks
        ]
        await async_bulk(self._client, actions)

    @classmethod
    def _chunk_to_source(cls, chunk: KnowledgeChunk) -> dict:
        """将领域 chunk 转换为 Elasticsearch source 结构。"""
        return {
            "user_id": chunk.user_id,
            "source_type": chunk.source_type,
            "source_id": chunk.source_id,
            "doc_name": chunk.doc_name,
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "parent_id": chunk.parent_id,
            "content": chunk.content,
            "vector": chunk.vector,
            "tags": chunk.tags,
            "created_at": chunk.created_at.isoformat(),
        }

    async def _get_embedding_config(self, user_id: str):
        """读取当前用户的 embedding 配置，并在未配置密钥时给出业务错误。"""
        async with self._uow_factory() as uow:
            app_config = await uow.app_config.get_or_create_default(user_id)
        embedding_config = app_config.notebook_config.embedding_config
        if not embedding_config.api_key.strip():
            raise BadRequestError(msg="Notebook Embedding模型未配置")
        return embedding_config

    @classmethod
    def _normalize(cls, scores: dict[str, float]) -> dict[str, float]:
        """将一组召回分数归一化到 0-1 区间，便于加权融合。"""
        if not scores:
            return {}
        values = list(scores.values())
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return {key: 1 for key in scores}
        return {key: (value - lo) / (hi - lo) for key, value in scores.items()}

    async def _resolve_parent_content(
        self,
        user_id: str,
        child_source: dict,
    ) -> str:
        """子块命中时回显父块内容，让搜索结果保留更完整上下文。"""
        parent_id = child_source.get("parent_id")
        if not parent_id:
            return child_source.get("content", "")
        resp = await self._client.search(
            index=CHUNKS_INDEX,
            body={
                "size": 1,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": user_id}},
                            {"term": {"chunk_id": parent_id}},
                        ]
                    }
                },
            },
        )
        docs = resp["hits"]["hits"]
        if docs:
            return docs[0]["_source"].get("content", "")
        return child_source.get("content", "")

    @classmethod
    def _build_chunks_index_body(cls, vector_dims: int) -> dict:
        """构建知识库 chunk 索引 mapping，向量维度由部署配置统一控制。"""
        return {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "doc_name": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "tags": {"type": "keyword"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": vector_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "created_at": {"type": "date"},
                }
            },
        }
