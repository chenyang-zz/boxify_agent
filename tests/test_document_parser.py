import pytest

from app.domain.services.notebook.parser import DocumentParser


def test_parse_pdf_rejects_non_pdf_content_with_clear_error():
    with pytest.raises(ValueError, match="文件内容不是合法PDF"):
        DocumentParser.parse(".pdf", b"not a pdf")
