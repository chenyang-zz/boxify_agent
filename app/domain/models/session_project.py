from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionProject(BaseModel):
    """会话项目领域模型，用于侧边栏扁平分组。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    name: str
    sort_order: int = 0
    is_pinned: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
