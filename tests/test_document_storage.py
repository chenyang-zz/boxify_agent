from app.infrastructure.external.document_storage.document_storage import (
    CosDocumentStorage,
    _READ_CHUNK_SIZE,
)


def test_read_body_to_bytes_reads_until_eof():
    body = FakeStreamBody([b"a" * 3, b"b" * 2, b"c"])

    content = CosDocumentStorage._read_body_to_bytes(body)

    assert content == b"aaabbc"
    assert body.read_sizes == [
        _READ_CHUNK_SIZE,
        _READ_CHUNK_SIZE,
        _READ_CHUNK_SIZE,
        _READ_CHUNK_SIZE,
    ]


class FakeStreamBody:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.read_sizes = []

    def read(self, chunk_size=1024):
        self.read_sizes.append(chunk_size)
        if not self._chunks:
            return b""
        return self._chunks.pop(0)
