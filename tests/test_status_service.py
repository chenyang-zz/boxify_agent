import pytest

from app.infrastructure.external.health_checker.elasticsearch_health_checker import (
    ElasticsearchHealthChecker,
)
from app.infrastructure.external.health_checker.postgres_health_checker import (
    PostgresHealthChecker,
)
from app.infrastructure.external.health_checker.redis_health_checker import (
    RedisHealthChecker,
)
from app.interfaces.service_dependencies import get_status_service


@pytest.mark.anyio
async def test_elasticsearch_health_checker_returns_ok_when_ping_succeeds():
    checker = ElasticsearchHealthChecker(FakeElasticsearch(FakeElasticClient(True)))

    status = await checker.check()

    assert status.service == "elasticsearch"
    assert status.status == "ok"
    assert status.details == ""


@pytest.mark.anyio
async def test_elasticsearch_health_checker_returns_error_when_ping_fails():
    checker = ElasticsearchHealthChecker(FakeElasticsearch(FakeElasticClient(False)))

    status = await checker.check()

    assert status.service == "elasticsearch"
    assert status.status == "error"
    assert status.details == "Elasticsearch服务Ping失败"


@pytest.mark.anyio
async def test_elasticsearch_health_checker_returns_error_when_ping_raises():
    checker = ElasticsearchHealthChecker(
        FakeElasticsearch(FakeElasticClient(RuntimeError("boom")))
    )

    status = await checker.check()

    assert status.service == "elasticsearch"
    assert status.status == "error"
    assert status.details == "boom"


def test_status_service_dependency_includes_elasticsearch_checker():
    service = get_status_service(
        db_session=FakeDbSession(),
        redis_client=FakeRedisClient(),
        elasticsearch=FakeElasticsearch(FakeElasticClient(True)),
    )

    checker_types = {type(checker) for checker in service._checkers}

    assert checker_types == {
        PostgresHealthChecker,
        RedisHealthChecker,
        ElasticsearchHealthChecker,
    }


class FakeElasticsearch:
    def __init__(self, client) -> None:
        self.client = client


class FakeElasticClient:
    def __init__(self, result) -> None:
        self._result = result

    async def ping(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeDbSession:
    pass


class FakeRedisClient:
    pass
