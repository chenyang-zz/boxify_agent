import json

import pytest

from app.utils.json_utils import parse_json_object


@pytest.mark.anyio
async def test_parse_json_object_parses_text_object() -> None:
    parsed = await parse_json_object(FakeJSONParser(), '{"value": 1}', {})

    assert parsed == {"value": 1}


@pytest.mark.anyio
async def test_parse_json_object_serializes_non_string_content() -> None:
    parsed = await parse_json_object(FakeJSONParser(), {"value": "中文"}, {})

    assert parsed == {"value": "中文"}


@pytest.mark.anyio
async def test_parse_json_object_returns_default_for_non_object() -> None:
    default = {"items": []}

    parsed = await parse_json_object(FakeJSONParser(), '["not-object"]', default)

    assert parsed is default


@pytest.mark.anyio
async def test_parse_json_object_returns_default_on_parser_error() -> None:
    default = {"items": []}

    parsed = await parse_json_object(FailingJSONParser(), '{"items": []}', default)

    assert parsed is default


class FakeJSONParser:
    async def invoke(self, text: str, default_value=None):
        try:
            return json.loads(text)
        except Exception:
            return default_value


class FailingJSONParser:
    async def invoke(self, text: str, default_value=None):
        raise RuntimeError("boom")
