from app.utils.hash import content_digest


def test_content_digest_returns_short_sha256_prefix():
    assert content_digest(b"boxify") == "a7c0a5fd1702cf8a"
    assert len(content_digest(b"boxify")) == 16
    assert len(content_digest(b"boxify", length=8)) == 8
