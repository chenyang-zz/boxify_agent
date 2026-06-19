import logging
from datetime import datetime
from typing import AsyncGenerator, Callable

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.domain.external.llm import LLM
from app.domain.models.event import DoneEvent, ErrorEvent, Event, MessageEvent
from app.domain.models.session import SessionType
from app.domain.repositories.vow import IUnitOfWork

logger = logging.getLogger(__name__)


class ChatService:
    """普通聊天会话服务，不创建 Agent 任务或沙箱。"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: LLM,
        user_id: str,
    ) -> None:
        self._uow_factory = uow_factory
        self._llm = llm
        self._user_id = user_id

    async def chat(
        self,
        session_id: str,
        message: str | None = None,
        latest_event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> AsyncGenerator[Event, None]:
        """向普通聊天会话发送消息，并以现有 SSE 事件结构返回。"""
        try:
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id_for_user(session_id, self._user_id)
            if not session:
                raise NotFoundError("该会话不存在")
            if session.type != SessionType.CHAT:
                raise BadRequestError("该会话不是普通聊天会话")
            if not message:
                return

            user_event = MessageEvent(role="user", message=message)
            messages = self._build_llm_messages(session.events, user_message=message)
            async with self._uow_factory() as uow:
                await uow.session.update_latest_message(
                    session_id=session_id,
                    message=message,
                    timestamp=timestamp or datetime.now(),
                )
                await uow.session.add_event(session_id, user_event)

            response = await self._llm.invoke(messages)
            assistant_message = str(response.get("content") or "")
            assistant_event = MessageEvent(
                role="assistant",
                message=assistant_message,
            )
            done_event = DoneEvent()
            async with self._uow_factory() as uow:
                await uow.session.add_event(session_id, assistant_event)
                await uow.session.add_event(session_id, done_event)

            yield assistant_event
            yield done_event
        except Exception as e:
            logger.error("普通聊天会话[%s]对话出错: %s", session_id, e)
            error_event = ErrorEvent(error=str(e))
            try:
                async with self._uow_factory() as uow:
                    await uow.session.add_event(session_id, error_event)
            except Exception as add_err:
                logger.warning("普通聊天会话[%s]添加错误事件失败: %s", session_id, add_err)
            yield error_event

    @staticmethod
    def _build_llm_messages(
        events: list[Event],
        user_message: str,
    ) -> list[dict[str, str]]:
        """从会话事件中提取 LLM message 历史。"""
        messages = [
            {"role": event.role, "content": event.message}
            for event in events
            if isinstance(event, MessageEvent) and event.message
        ]
        messages.append({"role": "user", "content": user_message})
        return messages
