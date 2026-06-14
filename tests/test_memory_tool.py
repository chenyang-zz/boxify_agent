import pytest

from app.domain.models.long_term_memory import LongTermMemory
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
