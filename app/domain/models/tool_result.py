from typing import Generic, Optional, Self, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    """工具结果Domain模型"""

    success: bool = True  # 是否成功调用
    message: Optional[str] = None  # 额外的信息提示
    data: Optional[T] = None  # 工具的执行结果/数据

    @classmethod
    def from_sandbox(
        cls,
        code: int,
        msg: str,
        data: Optional[T],
        **kwargs,
    ) -> Self:
        """将从沙箱中返回的API数据转换成工具结果"""
        return cls(
            success=True if code < 300 else False,
            message=msg,
            data=data,
        )
