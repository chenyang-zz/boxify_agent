.PHONY: dev api celery

dev:
	$(MAKE) -j2 api celery

api:
	uvicorn app.main:app --reload

celery:
	celery -A app.celery_app.celery_app worker --loglevel=info
