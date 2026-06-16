from math import sqrt


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    """计算两个等长向量的余弦相似度。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def average_vector(vectors: list[list[float] | None]) -> list[float]:
    """计算一组同维非空向量的平均值。"""
    valid_vectors = [vector for vector in vectors if vector]
    if not valid_vectors:
        return []
    dims = len(valid_vectors[0])
    return [
        sum(vector[index] for vector in valid_vectors) / len(valid_vectors)
        for index in range(dims)
    ]
