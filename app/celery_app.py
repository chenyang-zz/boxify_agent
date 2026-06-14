from celery import Celery

from core.config import get_settings

settings = get_settings()

# Notebook 解析任务运行在独立 worker 进程中，FastAPI 只负责派发任务。
celery_app = Celery(
    "boxify_notebook",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.notebook.document_parse", "app.tasks.memory.extract"],
)
