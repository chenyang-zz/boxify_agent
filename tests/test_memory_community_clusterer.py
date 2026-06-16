import pytest

from app.domain.models.memory_graph import (
    CommunityVoteEntity,
    CommunityVoteNeighbor,
    MemoryCommunityClusterStats,
)
from app.domain.services.memory.community_clusterer import MemoryCommunityClusterer


@pytest.mark.anyio
async def test_memory_community_clusterer_runs_full_lpa_and_fallback_metadata():
    repository = FakeCommunityGraphRepository(
        entities=[
            CommunityVoteEntity(
                id="entity-user",
                user_id="user-a",
                name="用户",
                type="生命体",
                embedding=[1.0, 0.0],
            ),
            CommunityVoteEntity(
                id="entity-music",
                user_id="user-a",
                name="周杰伦",
                type="生命体",
                embedding=[0.9, 0.1],
            ),
        ],
        neighbors={
            "entity-user": [
                CommunityVoteNeighbor(id="entity-music", embedding=[0.9, 0.1])
            ],
            "entity-music": [
                CommunityVoteNeighbor(id="entity-user", embedding=[1.0, 0.0])
            ],
        },
    )
    clusterer = MemoryCommunityClusterer(user_id="user-a", graph_repository=repository)

    stats = await clusterer.cluster()

    assert stats == MemoryCommunityClusterStats(
        communities=1,
        assigned_entities=2,
        merged_communities=0,
        enhanced_communities=1,
    )
    assert len(set(repository.assignments.values())) == 1
    community_id = next(iter(repository.assignments.values()))
    assert repository.metadata_updates == [
        ("user-a", community_id, "用户、周杰伦", "包含实体：用户, 周杰伦")
    ]


@pytest.mark.anyio
async def test_memory_community_clusterer_incremental_uses_existing_neighbor_label():
    repository = FakeCommunityGraphRepository(
        has_communities=True,
        dialogue_entity_ids=["entity-new"],
        entities=[
            CommunityVoteEntity(
                id="entity-new",
                user_id="user-a",
                name="新歌",
                type="作品",
                embedding=[0.9, 0.1],
            )
        ],
        members_by_community={
            "community-music": [
                CommunityVoteEntity(
                    id="entity-old",
                    user_id="user-a",
                    name="周杰伦",
                    type="生命体",
                    embedding=[1.0, 0.0],
                    community_id="community-music",
                )
            ]
        },
        neighbors={
            "entity-new": [
                CommunityVoteNeighbor(
                    id="entity-old",
                    community_id="community-music",
                    embedding=[1.0, 0.0],
                )
            ]
        },
    )
    clusterer = MemoryCommunityClusterer(user_id="user-a", graph_repository=repository)

    stats = await clusterer.cluster(dialogue_id="dialogue-1")

    assert stats.assigned_entities == 1
    assert stats.communities == 1
    assert repository.dialogue_calls == [("user-a", "dialogue-1")]
    assert repository.assignments == {"entity-new": "community-music"}


@pytest.mark.anyio
async def test_memory_community_clusterer_skips_when_dialogue_has_no_entities():
    repository = FakeCommunityGraphRepository(has_communities=True, dialogue_entity_ids=[])
    clusterer = MemoryCommunityClusterer(user_id="user-a", graph_repository=repository)

    stats = await clusterer.cluster(dialogue_id="dialogue-empty")

    assert stats.assigned_entities == 0
    assert stats.skipped == "no_entities"
    assert repository.assignments == {}


@pytest.mark.anyio
async def test_memory_community_clusterer_continues_when_single_metadata_fails():
    repository = FakeCommunityGraphRepository(
        entities=[
            CommunityVoteEntity(
                id="entity-user",
                user_id="user-a",
                name="用户",
                type="生命体",
                embedding=[1.0, 0.0],
            )
        ]
    )
    clusterer = MemoryCommunityClusterer(
        user_id="user-a",
        graph_repository=repository,
        summarizer=ExplodingCommunitySummarizer(),
    )

    stats = await clusterer.cluster()

    assert stats.assigned_entities == 1
    assert stats.enhanced_communities == 0


class FakeCommunityGraphRepository:
    def __init__(
        self,
        *,
        has_communities=False,
        dialogue_entity_ids=None,
        entities=None,
        neighbors=None,
        members_by_community=None,
    ):
        self._has_communities = has_communities
        self._dialogue_entity_ids = dialogue_entity_ids
        self._entities = entities or []
        self._neighbors = neighbors or {}
        self.members_by_community = members_by_community or {}
        self.dialogue_calls = []
        self.assignments = {}
        self.metadata_updates = []
        self.upserted = []
        self.refreshed = []

    async def has_communities(self, user_id):
        assert user_id == "user-a"
        return self._has_communities

    async def dialogue_entity_ids(self, user_id, dialogue_id):
        self.dialogue_calls.append((user_id, dialogue_id))
        return self._dialogue_entity_ids or []

    async def community_vote_entities(self, user_id, entity_ids=None):
        assert user_id == "user-a"
        if entity_ids is None:
            return self._entities
        by_id = {entity.id: entity for entity in self._entities}
        return [by_id[entity_id] for entity_id in entity_ids if entity_id in by_id]

    async def community_vote_neighbors(self, user_id, entity_ids):
        assert user_id == "user-a"
        return {
            entity_id: self._neighbors.get(entity_id, [])
            for entity_id in entity_ids
        }

    async def upsert_community(self, user_id, community_id):
        self.upserted.append((user_id, community_id))

    async def assign_entity_community(self, user_id, entity_id, community_id):
        self.assignments[entity_id] = community_id

    async def refresh_community_member_count(self, user_id, community_id):
        self.refreshed.append((user_id, community_id))
        return len(
            [
                entity_id
                for entity_id, assigned_id in self.assignments.items()
                if assigned_id == community_id
            ]
        )

    async def community_members(self, user_id, community_id):
        if community_id in self.members_by_community:
            return self.members_by_community[community_id]
        names = {
            entity.id: entity
            for entity in [
                *self._entities,
                *[
                    member
                    for members in self.members_by_community.values()
                    for member in members
                ],
            ]
        }
        return [
            names[entity_id]
            for entity_id, assigned_id in self.assignments.items()
            if assigned_id == community_id and entity_id in names
        ]

    async def community_relationships(self, user_id, community_id):
        return []

    async def update_community_metadata(self, user_id, community_id, name, summary):
        self.metadata_updates.append((user_id, community_id, name, summary))

    async def list_communities(self, user_id):
        return []

    async def prune_empty_communities(self, user_id):
        return None


class ExplodingCommunitySummarizer:
    async def summarize(self, members, relationships):
        raise RuntimeError("summary failed")
