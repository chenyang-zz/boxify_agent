from app.domain.services.memory.ontology import (
    UNKNOWN_ENTITY_TYPE,
    UNKNOWN_PREDICATE,
    normalize_entity_type,
    normalize_predicate,
)


def test_memory_ontology_normalizes_entities_and_predicates():
    assert normalize_entity_type("生命体") == "生命体"
    assert normalize_entity_type("Person") == UNKNOWN_ENTITY_TYPE
    assert normalize_entity_type(None) == UNKNOWN_ENTITY_TYPE

    assert normalize_predicate("偏好") == "偏好"
    assert normalize_predicate("LIKES") == UNKNOWN_PREDICATE
    assert normalize_predicate(None) == UNKNOWN_PREDICATE
