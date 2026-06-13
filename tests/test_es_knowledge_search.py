from types import SimpleNamespace

import pytest

from app.infrastructure.external.knowledge_search.es_knowledge_search import (
    CHUNKS_INDEX,
    ESKnowledgeSearch,
)


@pytest.mark.anyio
async def test_ensure_index_continues_when_ik_plugin_installed():
    client = FakeElasticsearchClient(
        nodes={
            "node-a": {
                "name": "node-a",
                "plugins": [{"name": "analysis-ik"}],
            }
        },
        index_exists=False,
    )
    search = make_search(client)

    await search.ensure_index()

    assert client.indices.exists_calls == [CHUNKS_INDEX]
    assert client.indices.created_indices == [CHUNKS_INDEX]
    assert client.indices.created_bodies[0]["mappings"]["properties"]["content"] == {
        "type": "text",
        "analyzer": "ik_max_word",
        "search_analyzer": "ik_smart",
    }


@pytest.mark.anyio
async def test_ensure_index_raises_when_single_node_missing_ik_plugin():
    client = FakeElasticsearchClient(
        nodes={
            "node-a": {
                "name": "node-a",
                "plugins": [{"name": "analysis-pinyin"}],
            }
        }
    )
    search = make_search(client)

    with pytest.raises(RuntimeError) as exc_info:
        await search.ensure_index()

    message = str(exc_info.value)
    assert "analysis-ik" in message
    assert "elasticsearch-plugin install" in message
    assert "重启Elasticsearch" in message
    assert "node-a" in message
    assert client.indices.exists_calls == []


@pytest.mark.anyio
async def test_ensure_index_raises_when_any_node_missing_ik_plugin():
    client = FakeElasticsearchClient(
        nodes={
            "node-a": {
                "name": "node-a",
                "plugins": [{"name": "analysis-ik"}],
            },
            "node-b": {
                "name": "node-b",
                "plugins": [],
            },
        }
    )
    search = make_search(client)

    with pytest.raises(RuntimeError) as exc_info:
        await search.ensure_index()

    message = str(exc_info.value)
    assert "analysis-ik" in message
    assert "node-b" in message
    assert "node-a" not in message
    assert client.indices.exists_calls == []


def make_search(client):
    search = ESKnowledgeSearch.__new__(ESKnowledgeSearch)
    search._client = client
    search._settings = SimpleNamespace(notebook_embedding_dims=1024)
    return search


class FakeElasticsearchClient:
    def __init__(self, nodes: dict, index_exists: bool = True) -> None:
        self.nodes = FakeNodesClient(nodes)
        self.indices = FakeIndicesClient(index_exists)


class FakeNodesClient:
    def __init__(self, nodes: dict) -> None:
        self._nodes = nodes
        self.info_calls = []

    async def info(self, metric: str) -> dict:
        self.info_calls.append(metric)
        return {"nodes": self._nodes}


class FakeIndicesClient:
    def __init__(self, index_exists: bool) -> None:
        self._index_exists = index_exists
        self.exists_calls = []
        self.created_indices = []
        self.created_bodies = []

    async def exists(self, index: str) -> bool:
        self.exists_calls.append(index)
        return self._index_exists

    async def create(self, index: str, body: dict) -> None:
        self.created_indices.append(index)
        self.created_bodies.append(body)
