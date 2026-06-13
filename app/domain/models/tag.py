from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Tag(BaseModel):
    """知识库标签领域模型"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    name: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
