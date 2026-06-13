import pytest

from app.domain.models.document import Document
from app.tasks.notebook.document_parse import _ensure_content_matches_document


def test_parse_task_rejects_content_size_mismatch():
    document = Document(
        user_id="user-a",
        file_name="notes.pdf",
        file_key="key",
        file_ext=".pdf",
        file_size=100,
    )

    with pytest.raises(ValueError) as exc_info:
        _ensure_content_matches_document(document, b"x" * 12)

    message = str(exc_info.value)
    assert "文档原文件读取大小异常" in message
    assert "上传记录=100 bytes" in message
    assert "COS读取=12 bytes" in message
