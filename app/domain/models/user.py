from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class User(BaseModel):
    """用户领域模型"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    password_hash: str = Field(repr=False)
    is_active: bool = True
    is_admin: bool = False
    oauth_provider: str | None = None
    oauth_subject: str | None = None
    email: str | None = None
    avatar_url: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
