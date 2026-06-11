#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/6/8 21:36
@Author : chenyangzhao542@gmail.com
@File   : event.py
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Self, Type, Union, cast, get_args

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.event import (
    Event,
    MessageEvent,
    PlanEvent,
    StepEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.file import File
from app.domain.models.plan import ExecutionStatus


class BaseEventData(BaseModel):
    """基础事件数据"""

    event_id: Optional[str] = None  # 事件id
    created_at: datetime = Field(default_factory=datetime.now)  # 事件时间

    # pydantic v2写法，序列化时将datetime转换为时间戳
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: int(v.timestamp()),
        }
    )

    @classmethod
    def base_event_data(cls, event: Event) -> Dict[str, Any]:
        """类方法，用于将事件Domain模型转换成基础事件数据字典"""
        return {
            "event_id": event.id,
            "created_at": int(event.created_at.timestamp()),
        }

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """类方法，从事件Domain模型中构建基础事件数据"""
        return cls(
            **cls.base_event_data(event),
            **event.model_dump(mode="json", exclude={"id", "created_at", "type"}),
        )


class BaseSSEEvent(BaseModel):
    """基础流式事件数据类型"""

    event: str  # 事件类型
    data: BaseEventData  # 数据

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """类方法，将事件Domain模型转换成基础流式事件"""
        # 获取事件数据的类型，如果没有则使用基础事件数据BaseEventData
        data_class: Type[BaseEventData] = cls.__annotations__.get("data", BaseEventData)

        # 调用构造函数完成初始化
        return cls(
            event=event.type,
            data=data_class.from_event(event),
        )


class CommonEventData(BaseEventData):
    """通用事件数据，让结构允许填充额外的数据"""

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: int(v.timestamp()),
        },
        extra="allow",
    )


class CommonSSEEvent(BaseSSEEvent):
    """通用流式事件"""

    event: str
    data: CommonEventData


class MessageEventData(BaseEventData):
    """消息事件数据"""

    role: Literal["user", "assistant"] = "assistant"
    message: str = ""
    attachments: List[File] = Field(default_factory=list)


class MessageSSEEvent(BaseSSEEvent):
    """消息流式事件"""

    event: Literal["message"] = "message"
    data: MessageEventData

    @classmethod
    def from_event(cls, event: Event) -> Self:
        event = cast(MessageEvent, event)
        return cls(
            data=MessageEventData(
                **BaseEventData.base_event_data(event),
                role=event.role,
                message=event.message,
                attachments=event.attachments,
            )
        )


class TitleEventData(BaseEventData):
    """标题事件数据"""

    title: str = ""


class TitleSSEEvent(BaseSSEEvent):
    """标题流式事件"""

    event: Literal["title"] = "title"
    data: TitleEventData


class StepEventData(BaseEventData):
    """步骤事件数据"""

    id: str = ""  # 步骤id
    status: ExecutionStatus  # 步骤执行状态
    description: str = ""  # 步骤描述


class StepSSEEvent(BaseSSEEvent):
    """步骤流式事件"""

    event: Literal["step"] = "step"
    data: StepEventData

    @classmethod
    def from_event(cls, event: Event) -> Self:
        event = cast(StepEvent, event)
        return cls(
            data=StepEventData(
                **BaseEventData.base_event_data(event),
                id=event.step.id,
                status=event.step.status,
                description=event.step.description,
            )
        )


class PlanEventData(BaseEventData):
    """计划事件数据"""

    steps: List[StepEventData]


class PlanSSEEvent(BaseSSEEvent):
    """计划流式事件"""

    event: Literal["plan"] = "plan"
    data: PlanEventData

    @classmethod
    def from_event(cls, event: Event) -> Self:
        event = cast(PlanEvent, event)
        return cls(
            data=PlanEventData(
                **BaseEventData.base_event_data(event),
                steps=[
                    StepEventData(
                        **BaseEventData.base_event_data(event),
                        id=step.id,
                        status=step.status,
                        description=step.description,
                    )
                    for step in event.plan.steps
                ],
            ),
        )


class ToolEventData(BaseEventData):
    """工具事件数据"""

    tool_call_id: str = ""  # 工具调用id
    name: str = ""  # 工具箱名字
    status: ToolEventStatus  # 工具状态
    function: str  # 工具名字
    args: Dict[str, Any]  # 工具参数
    content: Optional[Any] = None  # 工具调用结果


class ToolSSEEvent(BaseSSEEvent):
    """工具流式事件"""

    event: Literal["tool"] = "tool"
    data: ToolEventData

    @classmethod
    def from_event(cls, event: Event) -> Self:
        event = cast(ToolEvent, event)
        return cls(
            data=ToolEventData(
                **BaseEventData.base_event_data(event),
                tool_call_id=event.tool_call_id,
                name=event.tool_name,
                status=event.status,
                function=event.function_name,
                args=event.function_args,
                content=event.tool_content,
            )
        )


class DoneSSEEvent(BaseSSEEvent):
    """停止流失事件"""

    event: Literal["done"] = "done"


class WaitingSSEEvent(BaseSSEEvent):
    """等待人类输入流式事件"""

    event: Literal["waiting"] = "waiting"


class ErrorEventData(BaseEventData):
    """错误事件诗句"""

    error: str


class ErrorSSEEvent(BaseSSEEvent):
    """错误流式事件"""

    event: Literal["error"] = "error"
    data: ErrorEventData


# 定义Agent流式事件类型集合
AgentSSEEvent = Union[
    CommonSSEEvent,
    MessageSSEEvent,
    TitleSSEEvent,
    StepSSEEvent,
    PlanSSEEvent,
    TitleSSEEvent,
    DoneSSEEvent,
    WaitingSSEEvent,
    ErrorSSEEvent,
]


@dataclass
class EventMapping:
    """事件映射数据类，用于存储事件映射信息，涵盖流式事件类型、数据类、事件类型字符串"""

    sse_event_class: Type[BaseSSEEvent]
    data_class: Type[BaseEventData]
    event_type: str


class EventMapper:
    """事件映射类，利用Python自身提供的自省机制，将业务逻辑中的Event转换成适合流式传输的AgentSSEEvent"""

    # 缓存映射
    _cache_mapping: Optional[Dict[str, EventMapping]] = None

    @staticmethod
    def _get_event_type_mapping() -> Dict[str, EventMapping]:
        """通过反射动态构建从事件类型字符串到AgentSSEEvent的映射"""

        # 判断缓存映射是否存在，如果存在则直接返回
        if EventMapper._cache_mapping is not None:
            return EventMapper._cache_mapping

        # 获取AgentSSEEvent的所有可能类
        sse_event_classes = get_args(AgentSSEEvent)
        mapping = {}

        # 循环遍历AgentSSEEvent可能的所有类逐个处理
        for sse_event_class in sse_event_classes:
            # 跳过基类
            if sse_event_class == BaseSSEEvent:
                continue

            # 检查类是否包含event属性
            if (
                hasattr(sse_event_class, "__annotations__")
                and "event" in sse_event_class.__annotations__
            ):
                # 提取事件字段
                event_field = sse_event_class.__annotations__["event"]

                # 提取事件的具体值(Literal的值)
                if hasattr(event_field, "__args__") and len(event_field.__args__) > 0:
                    event_type = event_field.__args__[0]

                    # 提取sse的载荷数据
                    data_class: Any = None
                    if (
                        hasattr(sse_event_class, "__annotations__")
                        and "data" in sse_event_class.__annotations__
                    ):
                        data_class = sse_event_class.__annotations__["data"]

                    # 构建并注册映射关系
                    mapping[event_type] = EventMapping(
                        sse_event_class=sse_event_class,
                        data_class=data_class,
                        event_type=event_type,
                    )

        # 更新类级缓存
        EventMapper._cache_mapping = mapping
        return mapping

    @staticmethod
    def event_to_sse_event(event: Event) -> AgentSSEEvent:
        """将领域事件转换成Agent流式事件模型"""

        # 获取事件映射表
        event_type_mapping = EventMapper._get_event_type_mapping()

        # 根据传递的事件获取映射类
        event_mapping = event_type_mapping.get(event.type)

        # 如果找到了类型映射则进行转换
        if event_mapping:
            sse_event = event_mapping.sse_event_class.from_event(event)
            return cast(AgentSSEEvent, sse_event)

        # 如果没有找到则使用通用类型
        return CommonSSEEvent.from_event(event)

    @staticmethod
    def events_to_sse_events(events: List[Event]) -> List[AgentSSEEvent]:
        """将领域事件模型列表转换为SSE流式事件列表"""
        return list(
            filter(
                lambda x: x is not None,
                [EventMapper.event_to_sse_event(event) for event in events],
            )
        )
