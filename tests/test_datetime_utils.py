from datetime import timezone

from app.utils.datetime import parse_optional_datetime


def test_parse_optional_datetime_returns_none_for_empty_values() -> None:
    assert parse_optional_datetime(None) is None
    assert parse_optional_datetime("") is None
    assert parse_optional_datetime("NULL") is None
    assert parse_optional_datetime(" null ") is None


def test_parse_optional_datetime_parses_iso_datetime() -> None:
    parsed = parse_optional_datetime("2026-06-16T09:00:00")

    assert parsed is not None
    assert parsed.isoformat() == "2026-06-16T09:00:00"


def test_parse_optional_datetime_parses_z_suffix_as_utc() -> None:
    parsed = parse_optional_datetime("2026-06-16T09:00:00Z")

    assert parsed is not None
    assert parsed.isoformat() == "2026-06-16T09:00:00+00:00"
    assert parsed.tzinfo == timezone.utc


def test_parse_optional_datetime_returns_none_for_invalid_text() -> None:
    assert parse_optional_datetime("not-a-date") is None
