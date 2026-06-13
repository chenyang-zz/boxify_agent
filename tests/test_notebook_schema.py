from app.domain.models.knowledge import KnowledgeSearchHit
from app.interfaces.schemas.notebook import KnowledgeSearchHitResponse


def test_search_hit_response_validates_domain_model():
    hit = KnowledgeSearchHit(
        chunk_id="chunk-a",
        content="matched",
        doc_name="notes.pdf",
        source_id="doc-a",
        source_type="document",
        score=0.6,
    )

    response = KnowledgeSearchHitResponse.model_validate(hit)

    assert response.model_dump() == {
        "chunk_id": "chunk-a",
        "content": "matched",
        "doc_name": "notes.pdf",
        "source_id": "doc-a",
        "source_type": "document",
        "score": 0.6,
    }
