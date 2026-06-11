from uuid import uuid4

from pydantic import BaseModel, Field


class File(BaseModel):
    """文件信息Domain模型，用于记录Human/Assistant上传or生成的文件"""

    id: str = Field(default_factory=lambda: str(uuid4()))  # 文件id
    filename: str = ""  # 文件名字
    filepath: str = ""  # 文件路径
    key: str = ""  # 腾讯云cos中的路径
    extension: str = ""  # 扩展名
    mime_type: str = ""  # mine-type类型
    size: int = 0  # 文件大小，单位为字节
