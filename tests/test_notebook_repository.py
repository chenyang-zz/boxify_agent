import pytest

from app.domain.models.document import Document, DocumentStatus


@pytest.mark.anyio
async def test_document_repository_filters_documents_by_user():
    repository = InMemoryDocumentRepository()
    user_a_document = Document(user_id="user-a", file_name="a.txt", file_key="a")
    user_b_document = Document(user_id="user-b", file_name="b.txt", file_key="b")
    await repository.save(user_a_document)
    await repository.save(user_b_document)

    docs, total = await repository.list_by_user("user-a", page=1, page_size=20)

    assert total == 1
    assert docs == [user_a_document]
    assert await repository.get_by_user("user-a", user_b_document.id) is None


@pytest.mark.anyio
async def test_document_status_helpers_mark_parsing_and_failed():
    document = Document(user_id="user-a", file_name="a.txt", file_key="a")

    document.mark_parsing()
    assert document.status == DocumentStatus.PARSING
    assert document.progress == 0.1

    document.mark_failed("x" * 600)
    assert document.status == DocumentStatus.FAILED
    assert len(document.error_msg) == 500


class InMemoryDocumentRepository:
    def __init__(self):
        self.documents = []

    async def save(self, document):
        self.documents.append(document)

    async def get_by_user(self, user_id: str, document_id: str):
        for document in self.documents:
            if document.user_id == user_id and document.id == document_id:
                return document
        return None

    async def list_by_user(self, user_id: str, page: int, page_size: int, tag=None):
        docs = [doc for doc in self.documents if doc.user_id == user_id]
        return docs[(page - 1) * page_size : page * page_size], len(docs)
