from difflib import SequenceMatcher

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.memory_graph import (
    EntityNode,
    MemoryEntityDedupCandidate,
    MemoryEntityDedupDecision,
    MemoryEntityDedupResult,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.services.prompts.memory import (
    DEDUP_ENTITY_PROMPT,
    DEDUP_ENTITY_SYSTEM_PROMPT,
)
from app.utils.collections import merge_unique_strings
from app.utils.json_utils import parse_json_object
from app.utils.vector import cosine_similarity


class MemoryEntityDeduplicator:
    """保守实体消歧器：规则初筛，LLM 精判，高置信度才合并。"""

    def __init__(
        self,
        llm: LLM | None = None,
        json_parser: JSONParser | None = None,
        similarity_threshold: float = 0.8,
        merge_confidence: float = 0.8,
        max_candidates: int = 20,
        enable_llm: bool = True,
    ) -> None:
        self._llm = llm
        self._json_parser = json_parser
        self._similarity_threshold = similarity_threshold
        self._merge_confidence = merge_confidence
        self._max_candidates = max_candidates
        self._enable_llm = enable_llm

    async def dedup_batch(
        self, entity_by_idx: dict[int, EntityNode]
    ) -> MemoryEntityDedupResult:
        """对本次 LLM 输出的实体做批内同名和模糊消歧。"""
        canonical_items: list[tuple[int, EntityNode]] = []
        final_by_idx: dict[int, EntityNode] = {}
        redirects: dict[str, str] = {}

        for idx, entity in entity_by_idx.items():
            exact = self._find_exact(entity, [item[1] for item in canonical_items])
            if exact:
                self._merge_into(exact, entity)
                final_by_idx[idx] = exact
                if entity.id != exact.id:
                    redirects[entity.id] = exact.id
                continue

            merged = False
            for canonical_idx, canonical in self._candidate_items(
                entity, canonical_items
            ):
                decision = await self._decide(
                    left=canonical,
                    right=entity,
                    left_idx=canonical_idx,
                    right_idx=idx,
                )
                if not self._should_merge(decision):
                    continue
                if decision.canonical_idx == idx:
                    self._merge_into(entity, canonical)
                    self._replace_canonical(
                        canonical_items,
                        final_by_idx,
                        old=canonical,
                        new_idx=idx,
                        new=entity,
                    )
                    redirects[canonical.id] = entity.id
                    final_by_idx[idx] = entity
                else:
                    self._merge_into(canonical, entity)
                    final_by_idx[idx] = canonical
                    if entity.id != canonical.id:
                        redirects[entity.id] = canonical.id
                merged = True
                break

            if not merged:
                canonical_items.append((idx, entity))
                final_by_idx[idx] = entity

        return self._result(final_by_idx, redirects)

    async def merge_with_graph(
        self,
        user_id: str,
        entity_by_idx: dict[int, EntityNode],
        graph_repository: MemoryGraphRepository,
    ) -> MemoryEntityDedupResult:
        """写图前把本次实体与已有同类型实体做同名和模糊融合。"""
        cache: dict[str, list[EntityNode]] = {}
        redirects: dict[str, str] = {}
        final_by_idx = dict(entity_by_idx)

        for idx, entity in self._unique_idx_entities(final_by_idx):
            if entity.type not in cache:
                cache[entity.type] = await graph_repository.list_entities_by_type(
                    user_id, entity.type
                )
            existing_entities = cache[entity.type]
            exact = self._find_exact(entity, existing_entities)
            if exact:
                old_id = entity.id
                self._reuse_existing(entity, exact)
                if old_id != entity.id:
                    redirects[old_id] = entity.id
                continue

            for existing in self._candidate_entities(entity, existing_entities):
                decision = await self._decide(
                    left=existing,
                    right=entity,
                    left_idx=0,
                    right_idx=idx,
                )
                if not self._should_merge(decision):
                    continue
                old_id = entity.id
                self._reuse_existing(entity, existing)
                if old_id != entity.id:
                    redirects[old_id] = entity.id
                break

        return self._result(final_by_idx, redirects)

    def _candidate_items(
        self,
        entity: EntityNode,
        canonical_items: list[tuple[int, EntityNode]],
    ) -> list[tuple[int, EntityNode]]:
        """从批内规范实体中筛出可能与当前实体重复的候选。"""
        return [
            (idx, candidate)
            for idx, candidate in canonical_items
            if self._is_candidate(candidate, entity)
        ][: self._max_candidates]

    def _candidate_entities(
        self, entity: EntityNode, existing_entities: list[EntityNode]
    ) -> list[EntityNode]:
        """从图数据库已有实体中筛出同类型且相似的候选。"""
        return [
            candidate
            for candidate in existing_entities
            if self._is_candidate(candidate, entity)
        ][: self._max_candidates]

    async def _decide(
        self,
        left: EntityNode,
        right: EntityNode,
        left_idx: int,
        right_idx: int,
    ) -> MemoryEntityDedupDecision:
        """调用 LLM 对候选实体对做最终同一性判断，异常时保守不合并。"""
        if not self._enable_llm or not self._llm or not self._json_parser:
            return MemoryEntityDedupDecision()
        candidate = self._candidate(left, right)
        try:
            response = await self._llm.invoke(
                messages=[
                    {"role": "system", "content": DEDUP_ENTITY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": DEDUP_ENTITY_PROMPT.format(
                            left_idx=left_idx,
                            right_idx=right_idx,
                            left_name=left.name,
                            left_type=left.type,
                            left_description=left.description or "",
                            right_name=right.name,
                            right_type=right.type,
                            right_description=right.description or "",
                            name_similarity=f"{candidate.name_similarity:.3f}",
                            embedding_similarity=(
                                f"{candidate.embedding_similarity:.3f}"
                            ),
                            name_contains=str(candidate.name_contains).lower(),
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            parsed = await parse_json_object(
                self._json_parser, response.get("content"), {}
            )
            return MemoryEntityDedupDecision.model_validate(parsed)
        except Exception:
            return MemoryEntityDedupDecision()

    def _is_candidate(self, left: EntityNode, right: EntityNode) -> bool:
        """使用类型、名称相似度、向量相似度和包含关系做规则初筛。"""
        if left.type != right.type:
            return False
        candidate = self._candidate(left, right)
        return (
            candidate.name_similarity >= self._similarity_threshold
            or candidate.embedding_similarity >= self._similarity_threshold
            or candidate.name_contains
        )

    def _candidate(
        self, left: EntityNode, right: EntityNode
    ) -> MemoryEntityDedupCandidate:
        """构建实体对的相似度快照，供规则和 LLM 判断共用。"""
        left_name = self._norm_name(left.name)
        right_name = self._norm_name(right.name)
        name_contains = bool(
            left_name
            and right_name
            and (left_name in right_name or right_name in left_name)
        )
        return MemoryEntityDedupCandidate(
            left=left,
            right=right,
            name_similarity=SequenceMatcher(None, left_name, right_name).ratio(),
            embedding_similarity=cosine_similarity(left.embedding, right.embedding),
            name_contains=name_contains,
        )

    def _should_merge(self, decision: MemoryEntityDedupDecision) -> bool:
        """根据 LLM 决策和置信度阈值判断是否允许合并。"""
        return bool(
            decision.same_entity and decision.confidence >= self._merge_confidence
        )

    def _find_exact(
        self, entity: EntityNode, candidates: list[EntityNode]
    ) -> EntityNode | None:
        """按归一化名称和实体类型查找严格同名实体。"""
        norm_name = self._norm_name(entity.name)
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.type == entity.type
                and self._norm_name(candidate.name) == norm_name
            ),
            None,
        )

    @staticmethod
    def _merge_into(canonical: EntityNode, duplicate: EntityNode) -> None:
        """把重复实体的统计、画像和描述信息合并到规范实体。"""
        canonical.mention_count += duplicate.mention_count
        canonical.access_count = max(canonical.access_count, duplicate.access_count)
        canonical.importance = max(canonical.importance, duplicate.importance)
        canonical.confidence = max(canonical.confidence, duplicate.confidence)
        if duplicate.last_access_at and (
            not canonical.last_access_at
            or duplicate.last_access_at > canonical.last_access_at
        ):
            canonical.last_access_at = duplicate.last_access_at
        if duplicate.memory_layer == "long_term":
            canonical.memory_layer = "long_term"
        if len(duplicate.description) > len(canonical.description):
            canonical.description = duplicate.description
        if not canonical.embedding and duplicate.embedding:
            canonical.embedding = duplicate.embedding
        canonical.core_facts = merge_unique_strings(
            canonical.core_facts, duplicate.core_facts
        )
        canonical.traits = merge_unique_strings(canonical.traits, duplicate.traits)

    @staticmethod
    def _reuse_existing(entity: EntityNode, existing: EntityNode) -> None:
        """将新实体重定向为图中已有实体，并保留更完整的新旧信息。"""
        incoming_mentions = entity.mention_count
        old_description = entity.description
        old_embedding = entity.embedding
        old_importance = entity.importance
        old_confidence = entity.confidence
        entity.id = existing.id
        entity.name = existing.name
        entity.description = existing.description
        entity.embedding = existing.embedding or old_embedding
        entity.mention_count = existing.mention_count + incoming_mentions
        entity.access_count = existing.access_count
        entity.last_access_at = existing.last_access_at
        entity.memory_layer = (
            "long_term"
            if existing.memory_layer == "long_term"
            or entity.memory_layer == "long_term"
            else existing.memory_layer
        )
        entity.core_facts = merge_unique_strings(existing.core_facts, entity.core_facts)
        entity.traits = merge_unique_strings(existing.traits, entity.traits)
        entity.importance = max(old_importance, existing.importance)
        entity.confidence = max(old_confidence, existing.confidence)
        if len(old_description) > len(entity.description):
            entity.description = old_description

    @staticmethod
    def _replace_canonical(
        canonical_items: list[tuple[int, EntityNode]],
        entity_by_idx: dict[int, EntityNode],
        old: EntityNode,
        new_idx: int,
        new: EntityNode,
    ) -> None:
        """当新实体成为规范实体时，替换批内索引和规范列表引用。"""
        for index, (idx, entity) in enumerate(canonical_items):
            if entity is old:
                canonical_items[index] = (new_idx, new)
                break
        for idx, entity in list(entity_by_idx.items()):
            if entity is old:
                entity_by_idx[idx] = new

    def _result(
        self,
        entity_by_idx: dict[int, EntityNode],
        redirects: dict[str, str],
    ) -> MemoryEntityDedupResult:
        """收敛最终实体集合、索引映射和 ID 重定向表。"""
        unique_by_id: dict[str, EntityNode] = {}
        for entity in entity_by_idx.values():
            if entity.id not in unique_by_id:
                unique_by_id[entity.id] = entity
                continue
            if unique_by_id[entity.id] is entity:
                continue
            self._merge_into(unique_by_id[entity.id], entity)
        remapped = {
            idx: unique_by_id[entity.id]
            for idx, entity in entity_by_idx.items()
            if entity.id in unique_by_id
        }
        return MemoryEntityDedupResult(
            entities=list(unique_by_id.values()),
            entity_by_idx=remapped,
            redirects=redirects,
        )

    @staticmethod
    def _unique_idx_entities(
        entity_by_idx: dict[int, EntityNode],
    ) -> list[tuple[int, EntityNode]]:
        """按对象身份去重索引映射，避免同一实体被重复融合。"""
        seen: set[int] = set()
        unique: list[tuple[int, EntityNode]] = []
        for idx, entity in entity_by_idx.items():
            identity = id(entity)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append((idx, entity))
        return unique

    @staticmethod
    def _norm_name(name: str) -> str:
        """统一实体名称比较口径。"""
        return name.strip().lower()
