import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.bootstrap.memory import ensure_memory_graph_schema
from app.bootstrap.notebook import ensure_knowledge_index
from app.infrastructure.logging import setup_logging
from app.infrastructure.storage.cos import get_cos
from app.infrastructure.storage.elasticsearch import get_elasticsearch
from app.infrastructure.storage.neo4j import get_neo4j
from app.infrastructure.storage.postgres import get_postgres
from app.infrastructure.storage.redis import get_redis
from app.interfaces.endpoints.routes import router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.service_dependencies import (
    get_agent_task_service,
    get_app_config_bootstrap_service,
    get_auth_service,
)
from core.config import get_settings

# 1.加载配置信息
settings = get_settings()

# 2.初始化日志系统
setup_logging()
logger = logging.getLogger()

# 3. 定义FastAPI路由tags标签
openai_tags = [
    {
        "name": "认证模块",
        "description": "包含 **用户登录**、**当前用户信息** 等接口，用于完成后台访问鉴权。",
    },
    {
        "name": "状态模块",
        "description": "包含 **状态检测** 等 API 接口，用于检测系统的运行状态。",
    },
    {
        "name": "设置模块",
        "description": "包含 **LLM配置**、**Agent配置**、**MCP服务**、**A2A服务** 等接口，用于管理智能体运行配置。",
    },
    {
        "name": "文件模块",
        "description": "包含 **文件上传**、**文件信息查询**、**文件下载** 等接口，用于管理对话相关文件。",
    },
    {
        "name": "会话模块",
        "description": "包含 **会话创建**、**会话列表**、**会话详情**、**聊天流式响应**、**沙箱文件/终端/VNC** 等接口，用于管理智能体任务会话。",
    },
    {
        "name": "Notebook文档模块",
        "description": "包含 **文档上传**、**网页导入**、**文档列表/详情/状态**、**重试/删除**、**知识库检索** 等接口，用于管理用户独立知识库文档。",
    },
    {
        "name": "Notebook标签模块",
        "description": "包含 **标签列表** 等接口，用于管理和查询当前用户知识库文档标签。",
    },
    {
        "name": "记忆模块",
        "description": "包含 **主动记住**、**记忆列表**、**记忆检索**、**删除记忆** 等接口，用于管理用户长期记忆。",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """创建FastAPI应用程序生命周期上下文管理"""

    # 1.打印日志表示程序开始
    logger.info("Boxify正在初始化")

    # 2. 初始化Redis客户端
    await get_redis().init()
    await get_postgres().init()
    await get_cos().init()
    await get_elasticsearch().init()
    await get_neo4j().init()
    await ensure_knowledge_index()
    await ensure_memory_graph_schema()

    auth_service = get_auth_service()
    created_admin = await auth_service.bootstrap_admin(
        settings.admin_username,
        settings.admin_password,
    )
    if created_admin:
        logger.info("已初始化管理员用户")
    elif not settings.admin_username or not settings.admin_password:
        logger.warning("未配置管理员初始化账号或密码，跳过管理员初始化")
    admin_user = created_admin
    if not admin_user and settings.admin_username:
        admin_user = await auth_service.get_user_by_username(settings.admin_username)
    if admin_user:
        await get_app_config_bootstrap_service().bootstrap_admin_app_config(admin_user)

    try:
        # 3.lifespan节点/分解
        yield
    finally:
        # 4. 关闭
        logger.info("boxify正在关闭")
        try:
            # 等待agent服务关闭
            await asyncio.wait_for(get_agent_task_service().shutdown(), timeout=30.0)
            logger.info("agent服务成功关闭")
        except asyncio.TimeoutError:
            logger.warning("agent服务关闭超时，强制关闭，部分任务将被释放")
        except Exception as e:
            logger.error(f"agent服务关闭期间出现错误: {str(e)}")

        # 关闭其他应用
        await get_redis().shutdown()
        await get_postgres().shutdown()
        await get_cos().shutdown()
        await get_elasticsearch().shutdown()
        await get_neo4j().shutdown()


# 4.创建应用实例
app = FastAPI(
    title="Boxify通用智能体",
    description="Boxify是一个通用的AI Agent系统，可以完全私有部署，使用A2A+MCP连接Agent/Tool，同时支持在沙箱中运行各种内置工具和操作。",
    lifespan=lifespan,
    openapi_tags=openai_tags,
    version="1.0.0",
)

# 5.配置CROS中间件，解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "wails://localhost:9245",
        "http://localhost:9245",
        "https://localhost:9245",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6.注册错误处理器
register_exception_handlers(app)

# 7.集成路由
app.include_router(router, prefix="/api")
