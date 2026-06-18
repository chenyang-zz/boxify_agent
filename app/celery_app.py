from celery import Celery
from celery.schedules import crontab

from core.config import get_settings

settings = get_settings()


def build_memory_beat_schedule(settings) -> dict:
    if not settings.memory_maintenance_enabled:
        return {}

    maintenance_schedule = crontab(
        hour=settings.memory_maintenance_hour,
        minute=settings.memory_maintenance_minute,
    )
    beat_schedule = {}
    if settings.memory_scheduled_consolidate_enabled:
        beat_schedule["memory-scheduled-consolidate"] = {
            "task": "app.tasks.memory_scheduled_consolidate",
            "schedule": maintenance_schedule,
        }
    if settings.memory_scheduled_cluster_enabled:
        beat_schedule["memory-scheduled-cluster"] = {
            "task": "app.tasks.memory_scheduled_cluster",
            "schedule": maintenance_schedule,
        }
    if settings.memory_scheduled_reflect_enabled:
        beat_schedule["memory-scheduled-reflect"] = {
            "task": "app.tasks.memory_scheduled_reflect",
            "schedule": maintenance_schedule,
        }
    return beat_schedule


# Notebook 解析任务运行在独立 worker 进程中，FastAPI 只负责派发任务。
celery_app = Celery(
    "boxify_notebook",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.notebook.document_parse",
        "app.tasks.memory.extract",
        "app.tasks.memory.cluster",
        "app.tasks.memory.consolidate",
        "app.tasks.memory.reflect",
        "app.tasks.memory.scheduled",
    ],
)
celery_app.conf.beat_schedule = build_memory_beat_schedule(settings)
