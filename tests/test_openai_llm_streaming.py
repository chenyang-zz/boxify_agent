from types import SimpleNamespace

import pytest

from app.application.errors.exceptions import ServerRequestsError
from app.domain.models.app_config import LLMConfig
from app.infrastructure.external.llm.openai_llm import OpenAILLM


@pytest.mark.anyio
async def test_openai_llm_invoke_keeps_full_response_by_default():
    llm = OpenAILLM(LLMConfig(api_key="test-key"))
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _completion("hello")

    llm._client = _client_with_create(create)

    response = await llm.invoke([{"role": "user", "content": "hello"}])

    assert response == {"role": "assistant", "content": "hello"}
    assert "stream" not in calls[0]


@pytest.mark.anyio
async def test_openai_llm_stream_method_yields_text_chunks_and_skips_empty_deltas():
    llm = OpenAILLM(LLMConfig(api_key="test-key"))
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _stream(
            _chunk("he"),
            _chunk(None),
            _chunk("llo"),
        )

    llm._client = _client_with_create(create)

    chunks = llm.stream([{"role": "user", "content": "hello"}])

    assert [chunk async for chunk in chunks] == ["he", "llo"]
    assert calls[0]["stream"] is True
    assert "tools" not in calls[0]


@pytest.mark.anyio
async def test_openai_llm_streaming_errors_are_wrapped():
    llm = OpenAILLM(LLMConfig(api_key="test-key"))

    async def create(**kwargs):
        return _failing_stream(RuntimeError("boom"))

    llm._client = _client_with_create(create)

    chunks = llm.stream([{"role": "user", "content": "hello"}])

    with pytest.raises(ServerRequestsError):
        _ = [chunk async for chunk in chunks]


def _client_with_create(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def _chunk(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
            )
        ]
    )


def _completion(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    model_dump=lambda: {"role": "assistant", "content": content},
                ),
            )
        ],
        model_dump=lambda: {"choices": [{"message": {"content": content}}]},
    )


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


async def _failing_stream(error):
    raise error
    yield
