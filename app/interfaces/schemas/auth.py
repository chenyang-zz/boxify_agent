from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录请求"""

    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=255)


class UserResponse(BaseModel):
    """用户响应"""

    id: str
    username: str
    is_active: bool
    is_admin: bool


class LoginResponse(BaseModel):
    """登录响应"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
