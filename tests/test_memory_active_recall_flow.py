import pytest

from app.domain.models.app_config import AgentConfig
from app.domain.models.message import Message
from app.domain.services.flows.planner_react import PlannerReActFlow


@pytest.mark.anyio
async def test_planner_react_flow_prefixes_message_with_active_recall_context():
    flow = PlannerReActFlow(
        uow_factory=lambda: None,
        llm=object(),
        agent_config=AgentConfig(),
        session_id="session-1",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=FakeTool(),
        a2a_tool=FakeTool(),
        active_recall=FakeActiveRecall("【关于用户的已知信息】\n用户偏好华语流行音乐。"),
    )

    enriched = await flow._message_with_active_recall(
        Message(message="推荐一首歌", attachments=["file-1"])
    )

    assert enriched.message.startswith("【关于用户的已知信息】")
    assert "当前用户问题：\n推荐一首歌" in enriched.message
    assert enriched.attachments == ["file-1"]


@pytest.mark.anyio
async def test_planner_react_flow_keeps_original_message_when_recall_is_empty():
    flow = PlannerReActFlow(
        uow_factory=lambda: None,
        llm=object(),
        agent_config=AgentConfig(),
        session_id="session-1",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=FakeTool(),
        a2a_tool=FakeTool(),
        active_recall=FakeActiveRecall(""),
    )
    message = Message(message="推荐一首歌", attachments=["file-1"])

    assert await flow._message_with_active_recall(message) == message


class FakeActiveRecall:
    def __init__(self, context):
        self.context = context
        self.calls = []

    async def recall_context(self, query: str) -> str:
        self.calls.append(query)
        return self.context


class FakeTool:
    def get_tools(self):
        return []

    def has_tool(self, tool_name):
        return False
