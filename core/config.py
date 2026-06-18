from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Boxify后台中控配置信息，从.env或者环境变量中加载数据"""

    # 项目基础配置
    # 当前运行环境标识，用于区分 development、production 等部署环境
    env: str = "development"
    # 应用日志级别，影响全局日志输出详细程度
    log_level: str = "INFO"
    # 仅用于首次导入旧版 YAML 配置，运行时配置以数据库为准
    app_config_filepath: str = "config.yaml"

    # 数据库相关配置
    # PostgreSQL 异步连接地址，包含账号密码时禁止出现在 repr 和日志中
    sqlalchemy_database_uri: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/boxify",
        repr=False,
    )

    # Redis缓存配置
    # Redis 主机地址，用于缓存、消息流和部分异步任务协作
    redis_host: str = ""
    # Redis 监听端口
    redis_port: int = 6379
    # Redis 默认数据库编号
    redis_db: int = 0
    # Redis 访问密码，未设置时按无密码连接处理
    redis_password: str | None = Field(default=None, repr=False)

    # Cos腾讯云对象存储配置
    # 腾讯云 COS SecretId，用于签名访问对象存储
    cos_secret_id: str = Field(default="", repr=False)
    # 腾讯云 COS SecretKey，用于签名访问对象存储
    cos_secret_key: str = Field(default="", repr=False)
    # COS bucket 所在地域
    cos_region: str = ""
    # COS 访问协议，通常为 https
    cos_scheme: str = "https"
    # COS bucket 名称
    cos_bucket: str = ""
    # COS 自定义访问域名，为空时使用默认域名
    cos_domain: str = ""

    # Auth配置
    # JWT 签名密钥，生产环境必须替换为强随机值
    auth_secret_key: str = Field(default="change-me-in-development", repr=False)
    # Access token 有效期，单位为分钟
    auth_access_token_expire_minutes: int = 1440
    # 首次启动时初始化管理员账号的用户名
    admin_username: str = ""
    # 首次启动时初始化管理员账号的密码，只用于 bootstrap
    admin_password: str = Field(default="", repr=False)
    # 数据库中的用户独立应用配置加密密钥，需在生产环境保持稳定
    app_config_encryption_key: str = Field(default="", repr=False)

    # Notebook 检索向量维度，需与 Elasticsearch 索引和 embedding 模型输出一致
    notebook_embedding_dims: int = 1024

    # Neo4j 记忆图谱配置
    # 用于长期记忆实体、关系、反思结果和主动召回上下文的图数据库
    # Neo4j Bolt 连接地址
    neo4j_uri: str = "bolt://localhost:7687"
    # Neo4j 登录用户名
    neo4j_username: str = "neo4j"
    # Neo4j 登录密码，禁止出现在 repr 和日志中
    neo4j_password: str = Field(default="neo4j-password", repr=False)
    # Neo4j database 名称
    neo4j_database: str = "neo4j"

    # 短期记忆提升为长期记忆的阈值，以及画像增强时处理的长期实体数量
    # 短期记忆被访问达到该次数后才允许提升为长期记忆
    memory_consolidate_min_access: int = 3
    # 短期记忆重要性达到该分数后才允许提升为长期记忆
    memory_consolidate_min_importance: float = 0.8
    # 短期记忆被提及达到该次数后才允许提升为长期记忆
    memory_consolidate_min_mention: int = 3
    # 短期记忆创建超过该小时数后才参与长期记忆巩固
    memory_consolidate_min_age_hours: int = 24
    # 每次巩固时最多增强画像的长期实体数量
    memory_consolidate_profile_top_k: int = 20

    # 记忆反思触发条件、候选实体数量、陈述采样量和生成洞察数量限制
    # 新增记忆累计达到该数量后触发反思流程
    memory_reflection_trigger_threshold: int = 10
    # 反思时最多选取的候选实体数量
    memory_reflection_top_k: int = 30
    # 候选实体少于该数量时跳过反思
    memory_reflection_min_entities: int = 5
    # 每个实体参与反思的陈述采样数量
    memory_reflection_stmt_per_entity: int = 5
    # 单次反思至少保留的洞察数量
    memory_reflection_min_insights: int = 1
    # 单次反思最多生成的洞察数量
    memory_reflection_max_insights: int = 5

    # Agent 对话时主动召回记忆的耗时、数量、分数阈值和上下文注入长度限制
    # 主动召回允许占用的最长时间，单位为秒
    memory_active_recall_timeout_seconds: float = 3.5
    # 主动召回时最多返回的实体记忆数量
    memory_active_recall_entity_top_k: int = 3
    # 主动召回时最多返回的反思洞察数量
    memory_active_recall_insight_top_k: int = 3
    # 主动召回结果进入上下文的最低相关性分数
    memory_active_recall_min_score: float = 0.72
    # 注入 Agent 上下文的主动召回文本最大字符数
    memory_active_recall_max_chars: int = 1200

    # 记忆社区聚类的 LPA 迭代、权重、合并阈值和元数据采样限制
    # LPA 最多迭代轮数
    memory_community_max_iterations: int = 10
    # 平均向量余弦大于该值时合并社区
    memory_community_merge_threshold: float = 0.85
    # 邻居语义相似度投票权重
    memory_community_semantic_weight: float = 0.6
    # 一跳关系连接投票权重
    memory_community_relation_weight: float = 0.4
    # 生成社区名称/摘要时最多采样的成员数量
    memory_community_metadata_member_limit: int = 20

    # 记忆自动维护调度配置，默认关闭以避免开发环境产生后台 LLM/Neo4j 成本
    # 是否启用 Celery beat 记忆维护任务
    memory_maintenance_enabled: bool = False
    # 是否定时派发记忆巩固任务
    memory_scheduled_consolidate_enabled: bool = False
    # 是否定时派发社区全量聚类任务
    memory_scheduled_cluster_enabled: bool = False
    # 是否定时派发记忆反思任务
    memory_scheduled_reflect_enabled: bool = False
    # 记忆维护任务每日运行小时
    memory_maintenance_hour: int = 4
    # 记忆维护任务每日运行分钟
    memory_maintenance_minute: int = 0

    # Elasticsearch配置
    # Elasticsearch HTTP 地址，用于 Notebook 知识库混合检索
    elasticsearch_url: str = "http://localhost:9200"
    # Elasticsearch 用户名，未启用安全认证时可为空
    elasticsearch_username: str = ""
    # Elasticsearch 密码，未启用安全认证时可为空
    elasticsearch_password: str = Field(default="", repr=False)
    # Notebook 知识库检索初始化、写入和查询 Elasticsearch 的请求超时
    elasticsearch_request_timeout: int = 30

    # Celery配置
    # Notebook 文档解析和记忆任务派发使用的消息队列与结果后端
    # Celery broker 地址，当前通常使用 Redis 数据库
    celery_broker_url: str = "redis://localhost:6379/1"
    # Celery result backend 地址，用于任务结果或状态存储
    celery_result_backend: str = "redis://localhost:6379/2"

    # Sandbox配置
    # 沙箱服务地址，为空时表示未启用远程沙箱
    sandbox_address: Optional[str] = None
    # 创建沙箱容器时使用的镜像名称
    sandbox_image: Optional[str] = None
    # 沙箱容器名称前缀
    sandbox_name_prefix: Optional[str] = None
    # 沙箱空闲或任务完成后的默认保留时长，单位为分钟
    sandbox_ttl_minutes: Optional[int] = 60
    # 沙箱容器加入的 Docker 网络
    sandbox_network: Optional[str] = None
    # 沙箱内 Chrome 启动参数
    sandbox_chrome_args: Optional[str] = ""
    # 沙箱内 HTTPS 代理配置
    sandbox_https_proxy: Optional[str] = None
    # 沙箱内 HTTP 代理配置
    sandbox_http_proxy: Optional[str] = None
    # 沙箱内不走代理的地址列表
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
