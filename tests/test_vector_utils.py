import pytest

from app.utils.vector import average_vector, cosine_similarity


def test_cosine_similarity_returns_one_for_same_direction():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_handles_invalid_vectors():
    assert cosine_similarity(None, [1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_average_vector_returns_component_average():
    assert average_vector([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]


def test_average_vector_ignores_empty_vectors():
    assert average_vector([None, [], [1.0, 3.0], [3.0, 5.0]]) == [2.0, 4.0]


def test_average_vector_returns_empty_when_no_valid_vectors():
    assert average_vector([None, []]) == []
