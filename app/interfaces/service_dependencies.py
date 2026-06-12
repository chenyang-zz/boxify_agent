import logging

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio.session import AsyncSession

from app.application.errors.exceptions import UnauthorizedError
from app.application.services.agent_service import AgentService
from app.application.services.agent_task_service import AgentTaskService
from app.application.services.app_config_bootstrap_service import (
    AppConfigBootstrapService,
)
from app.application.services.app_config_service import AppConfigService
from app.application.services.auth_service import AuthService
from app.application.services.file_service import FileService
from app.application.services.session_service import SessionService
from app.application.services.status_service import StatusService
from app.domain.models.user import User
from app.infrastructure.external.file_storage.cos_file_storage import CosFileStorage
from app.infrastructure.external.health_checker.postgres_health_checker import (
    PostgresHealthChecker,
)
from app.infrastructure.external.health_checker.redis_health_checker import (
    RedisHealthChecker,
)
from app.infrastructure.external.json_parser.repair_json_parser import RepairJSONParser
from app.infrastructure.external.llm.openai_llm import OpenAILLM
from app.infrastructure.external.task.redis_stream_task import RedisStreamTask
from app.infrastructure.repositories.file_app_config_repository import (
    FileAppConfigRepository,
)
from app.infrastructure.sandbox.docker_sandbox import DockerSandbox
from app.infrastructure.search.bing_search import BingSearchEngine
from app.infrastructure.storage.cos import Cos, get_cos
from app.infrastructure.storage.postgres import get_db_session, get_uow
from app.infrastructure.storage.redis import RedisClient, get_redis
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)
AGENT_TASK_CLS = RedisStreamTask


def get_auth_service() -> AuthService:
    """获取认证服务"""
    return AuthService(
        uow_factory=get_uow,
        secret_key=settings.auth_secret_key,
        access_token_expire_minutes=settings.auth_access_token_expire_minutes,
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """根据Bearer token获取当前用户"""
    if not credentials or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError()
    return await auth_service.get_user_by_token(credentials.credentials)


async def require_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> HTTPAuthorizationCredentials:
    """要求请求携带Bearer token，不访问数据库"""
    if not credentials or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError()
    return credentials


async def require_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """要求当前用户为可用状态"""
    if not current_user.is_active:
        raise UnauthorizedError()
    return current_user


def get_app_config_service(
    current_user: User = Depends(require_active_user),
) -> AppConfigService:
    """获取应用配置服务"""
    logger.info("加载获取AppConfigService")
    return AppConfigService(uow_factory=get_uow, user_id=current_user.id)


def get_app_config_bootstrap_service() -> AppConfigBootstrapService:
    """获取应用配置初始化服务"""
    return AppConfigBootstrapService(
        uow_factory=get_uow,
        legacy_app_config_repository=FileAppConfigRepository(
            settings.app_config_filepath
        ),
    )


def get_status_service(
    db_session: AsyncSession = Depends(get_db_session),
    redis_client: RedisClient = Depends(get_redis),
) -> StatusService:
    """获取状态服务"""
    # 1.初始化postgres和redis状态检查
    postgres_checker = PostgresHealthChecker(db_session)
    redis_checker = RedisHealthChecker(redis_client)

    # 2.创建服务并返回
    logger.info("加载获取StatusService")
    return StatusService(checkers=[postgres_checker, redis_checker])


def get_file_service(
    cos: Cos = Depends(get_cos),
) -> FileService:
    # 初始化文件仓库和文件存储桶
    file_storage = CosFileStorage(
        uow_factory=get_uow,
        bucket=settings.cos_bucket,
        cos=cos,
    )

    # 构建服务并返回
    return FileService(
        uow_factory=get_uow,
        file_storage=file_storage,
    )


def get_session_service() -> SessionService:
    return SessionService(uow_factory=get_uow, sandbox_cls=DockerSandbox)


def get_agent_task_service() -> AgentTaskService:
    """获取Agent任务生命周期服务"""
    return AgentTaskService(task_cls=AGENT_TASK_CLS)


async def get_agent_service(
    cos: Cos = Depends(get_cos),
    current_user: User = Depends(require_active_user),
) -> AgentService:
    # 获取当前用户应用配置信息(读取配置需要实时获取,所以不配置缓存)
    app_config = await AppConfigService(
        uow_factory=get_uow, user_id=current_user.id
    ).get_app_config()

    # 构建依赖实例
    llm = OpenAILLM(app_config.llm_config)
    file_storage = CosFileStorage(
        uow_factory=get_uow,
        bucket=settings.cos_bucket,
        cos=cos,
    )

    # 实例化Agent服务并返回
    return AgentService(
        uow_factory=get_uow,
        llm=llm,
        agent_config=app_config.agent_config,
        mcp_config=app_config.mcp_config,
        a2a_config=app_config.a2a_config,
        sandbox_cls=DockerSandbox,
        task_cls=AGENT_TASK_CLS,
        json_parser=RepairJSONParser(),
        search_engine=BingSearchEngine,
        file_storage=file_storage,
    )
