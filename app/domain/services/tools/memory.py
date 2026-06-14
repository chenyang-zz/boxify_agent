from app.domain.models.tool_result import ToolResult
from app.domain.services.memory import MemorySearch
from app.domain.services.tools.base import BaseTool, tool


class MemoryTool(BaseTool):
    """长期记忆工具包。"""

    name: str = "memory"

    def __init__(self, memory: MemorySearch) -> None:
        super().__init__()
        self._memory = memory

    @tool(
        name="search_memory",
        description="检索用户长期记忆。当问题涉及用户的个人信息、偏好、经历、关系或曾经说过的内容时使用。",
        parameters={
            "query": {
                "type": "string",
                "description": "用于检索用户长期记忆的关键词或短问题。",
            },
            "top_k": {
                "type": "integer",
                "description": "返回的记忆条数，默认5条，最多20条。",
            },
        },
        required=["query"],
    )
    async def search_memory(
        self, query: str, top_k: int = 5
    ) -> ToolResult[list[dict[str, object]]]:
        """检索长期记忆并返回给 Agent。"""
        memories = await self._memory.search(query, top_k)
        return ToolResult(
            success=True,
            message="检索记忆成功",
            data=[
                {
                    "id": memory.id,
                    "content": memory.content,
                    "summary": memory.summary,
                    "keywords": memory.keywords,
                }
                for memory in memories
            ],
        )
