from typing import TypeVar, Generic, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")


class Response(BaseModel, Generic[T]):
    """基础API相应结果，继承BaseModel，并定义范型"""

    code: int = 200  # 业务状态码，和http状态码保持一致
    msg: str = "success"  # 响应消息提示
    data: Optional[T] = Field(default_factory=dict)  # 响应数据默认为空字典

    @staticmethod
    def success(msg: str = "success", data: Optional[T] = None) -> "Response[T]":
        """成功消息，传递msg+data，code固定为200"""
        return Response(code=200, msg=msg, data=data if data is not None else {})

    @staticmethod
    def fail(code: int, msg: str, data: Optional[T] = None) -> "Response[T]":
        """失败消息，传递code+msg+data"""
        return Response(code=code, msg=msg, data=data if data is not None else {})
