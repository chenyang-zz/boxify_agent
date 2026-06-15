from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Boxify后台中控配置信息，从.env或者环境变量中加载数据"""

    # 项目基础配置
    env: str = "development"
    log_level: str = "INFO"
    app_config_filepath: str = "config.yaml"

    # 数据库相关配置
    sqlalchemy_database_uri: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/boxify",
        repr=False,
    )

    # Redis缓存配置
    redis_host: str = ""
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = Field(default=None, repr=False)

    # Cos腾讯云对象存储配置
    cos_secret_id: str = Field(default="", repr=False)
    cos_secret_key: str = Field(default="", repr=False)
    cos_region: str = ""
    cos_scheme: str = "https"
    cos_bucket: str = ""
    cos_domain: str = ""

    # Auth配置
    auth_secret_key: str = Field(default="change-me-in-development", repr=False)
    auth_access_token_expire_minutes: int = 1440
    admin_username: str = ""
    admin_password: str = Field(default="", repr=False)
    app_config_encryption_key: str = Field(default="", repr=False)

    notebook_embedding_dims: int = 1024

    # Neo4j 记忆图谱配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = Field(default="neo4j-password", repr=False)
    neo4j_database: str = "neo4j"
    memory_consolidate_min_access: int = 3
    memory_consolidate_min_importance: float = 0.8
    memory_consolidate_min_mention: int = 3
    memory_consolidate_min_age_hours: int = 24
    memory_consolidate_profile_top_k: int = 20

    # Elasticsearch配置
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_username: str = ""
    elasticsearch_password: str = Field(default="", repr=False)
    elasticsearch_request_timeout: int = 30

    # Celery配置
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Sandbox配置
    sandbox_address: Optional[str] = None
    sandbox_image: Optional[str] = None
    sandbox_name_prefix: Optional[str] = None
    sandbox_ttl_minutes: Optional[int] = 60
    sandbox_network: Optional[str] = None
    sandbox_chrome_args: Optional[str] = ""
    sandbox_https_proxy: Optional[str] = None
    sandbox_http_proxy: Optional[str] = None
    sandbox_no_proxy: Optional[str] = None

    # 使用 pydantic v2的写法来完成环境变量的告知
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """获取当前项目的配置信息，并对内容进行缓存，避免重复读取"""
    settings = Settings()
    return settings
