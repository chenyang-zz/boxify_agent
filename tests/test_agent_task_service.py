import pytest

from app.application.services.agent_task_service import AgentTaskService


@pytest.mark.anyio
async def test_agent_task_service_shutdown_delegates_to_task_class():
    FakeTask.destroy_calls = 0
    service = AgentTaskService(task_cls=FakeTask)

    await service.shutdown()

    assert FakeTask.destroy_calls == 1


class FakeTask:
    destroy_calls = 0

    @classmethod
    async def destroy(cls):
        cls.destroy_calls += 1
