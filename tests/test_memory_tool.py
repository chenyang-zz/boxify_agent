import pytest

from app.domain.models.long_term_memory import LongTermMemory
from app.domain.models.memory_graph import GraphRelationFact, LongTermMemoryGraphData
from app.domain.services.tools.memory import MemoryTool


@pytest.mark.anyio
async def test_memory_tool_searches_user_long_term_memories():
    memory = FakeMemorySearch(
        [
            LongTermMemory(
                user_id="user-a",
                content="我喜欢周杰伦的歌",
                summary="用户喜欢周杰伦",
                keywords=["周杰伦"],
                graph_data=LongTermMemoryGraphData(
                    entity_id="entity-1",
                    entity_name="周杰伦",
                    entity_type="Person",
                    description="歌手",
                    importance=0.9,
                    memory_layer="long_term",
                    core_facts=["用户长期喜欢周杰伦"],
                    traits=["偏好华语流行"],
                    access_count=3,
                    mention_count=4,
                    relations=[
                        GraphRelationFact(
                            name="LIKES",
                            direction="incoming",
                            neighbor_name="用户",
                            neighbor_type="Person",
                            evidence="用户喜欢周杰伦的歌",
                        )
                    ],
                    source_memory_summary="用户喜欢周杰伦",
                ),
            )
        ]
    )
    tool = MemoryTool(memory)

    result = await tool.search_memory("喜欢的歌手", top_k=3)

    assert result.success is True
    assert result.data == [
        {
            "id": memory.memories[0].id,
            "content": "我喜欢周杰伦的歌",
            "summary": "用户喜欢周杰伦",
            "keywords": ["周杰伦"],
            "graph": {
                "entity_id": "entity-1",
                "entity_name": "周杰伦",
                "entity_type": "Person",
                "description": "歌手",
                "score": 0,
                "importance": 0.9,
                "memory_layer": "long_term",
                "core_facts": ["用户长期喜欢周杰伦"],
                "traits": ["偏好华语流行"],
                "access_count": 3,
                "mention_count": 4,
                "relations": [
                    {
                        "name": "LIKES",
                        "direction": "incoming",
                        "neighbor_name": "用户",
                        "neighbor_type": "Person",
                        "evidence": "用户喜欢周杰伦的歌",
                        "is_current": True,
                    }
                ],
                "source_memory_summary": "用户喜欢周杰伦",
            },
        }
    ]
    assert memory.search_calls == [("喜欢的歌手", 3)]


class FakeMemorySearch:
    def __init__(self, memories):
        self.memories = memories
        self.search_calls = []

    async def search(self, query: str, top_k: int):
        self.search_calls.append((query, top_k))
        return self.memories
