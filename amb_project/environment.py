"""Post/comment section logic, seed generation, and simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from agents import Agent, Opinion, create_population


class Condition(Enum):
    CONTROL = "control"
    TREATMENT1 = "treatment1"
    TREATMENT2 = "treatment2"
    TREATMENT3 = "treatment3"
    FAVOR_B_CONTROL = "favor_b_control"
    FAVOR_B_TREATMENT1 = "favor_b_treatment1"
    FAVOR_B_TREATMENT2 = "favor_b_treatment2"
    FAVOR_B_TREATMENT3 = "favor_b_treatment3"


class SeedBias(Enum):
    NEUTRAL = "neutral"
    FAVOR_A = "favor_a"
    FAVOR_B = "favor_b"


NUM_POSTS = 40
SEED_COUNT = 20


def get_seed_bias(condition: Condition, post_index: int) -> SeedBias:
    """Return seed bias for a 1-based post index."""
    if condition in (Condition.CONTROL, Condition.FAVOR_B_CONTROL):
        return SeedBias.NEUTRAL

    if condition in (Condition.TREATMENT1, Condition.TREATMENT2, Condition.TREATMENT3):
        if post_index <= 20:
            return SeedBias.FAVOR_A
        if condition is Condition.TREATMENT1:
            return SeedBias.NEUTRAL
        if condition is Condition.TREATMENT2:
            return SeedBias.FAVOR_B
        return SeedBias.FAVOR_A

    if condition in (
        Condition.FAVOR_B_TREATMENT1,
        Condition.FAVOR_B_TREATMENT2,
        Condition.FAVOR_B_TREATMENT3,
    ):
        if post_index <= 20:
            return SeedBias.FAVOR_B
        if condition is Condition.FAVOR_B_TREATMENT1:
            return SeedBias.NEUTRAL
        if condition is Condition.FAVOR_B_TREATMENT2:
            return SeedBias.FAVOR_A
        return SeedBias.FAVOR_B

    raise ValueError(f"Unknown condition: {condition}")


def generate_seed_comments(
    bias: SeedBias, rng: np.random.Generator, n: int = SEED_COUNT
) -> list[Opinion]:
    """Return seed comment opinions for a post."""
    if bias is SeedBias.NEUTRAL:
        prob_a = 0.5
    elif bias is SeedBias.FAVOR_A:
        prob_a = 0.7
    else:
        prob_a = 0.3

    return [Opinion.A if rng.random() < prob_a else Opinion.B for _ in range(n)]


@dataclass
class Comment:
    source: str
    expressed: Opinion
    is_conforming: bool
    agent_id: Optional[int] = None


@dataclass
class AgentStateSnapshot:
    agent_id: int
    private_opinion: Opinion
    opinion_confidence: float
    willingness_to_comment: float
    silence_streak: int


@dataclass
class PostRecord:
    post_index: int
    comments: list[Comment] = field(default_factory=list)
    commented_agent_ids: set[int] = field(default_factory=set)
    arrival_support: dict[int, float] = field(default_factory=dict)
    agent_snapshots: list[AgentStateSnapshot] = field(default_factory=list)


class Simulation:
    """Run one experimental condition across 40 posts."""

    def __init__(
        self,
        condition: Condition,
        rng: np.random.Generator,
        n_agents: int = 200,
        trait_overrides: Optional[dict[str, float]] = None,
        enable_opinion_flip: bool = False,
        agents: Optional[list[Agent]] = None,
    ) -> None:
        self.condition = condition
        self.rng = rng
        self.enable_opinion_flip = enable_opinion_flip
        self.agents = (
            agents
            if agents is not None
            else create_population(rng, n_agents=n_agents, trait_overrides=trait_overrides)
        )
        self.history: list[PostRecord] = []

    def run(self) -> list[PostRecord]:
        if not self.agents:
            return []

        for post_index in range(1, NUM_POSTS + 1):
            self._run_post(post_index)
        return self.history

    def _run_post(self, post_index: int) -> None:
        bias = get_seed_bias(self.condition, post_index)
        seed_opinions = generate_seed_comments(bias, self.rng)

        comments: list[Comment] = [
            Comment(source="seed", expressed=opinion, is_conforming=False)
            for opinion in seed_opinions
        ]
        n_a = sum(1 for opinion in seed_opinions if opinion is Opinion.A)
        n_b = len(seed_opinions) - n_a

        arrival_order = self.rng.permutation(len(self.agents))
        arrival_support: dict[int, float] = {}
        commented_ids: set[int] = set()

        for idx in arrival_order:
            agent = self.agents[idx]

            # Record support_for_self at arrival, before this agent's comment
            # decision and before any subsequent agents have commented.
            support = agent.visible_support_for_self(n_a, n_b)
            arrival_support[agent.agent_id] = support

            if agent.decide_to_comment(support, self.rng):
                expressed, is_conforming = agent.decide_expression(support, self.rng)
                comments.append(
                    Comment(
                        source="agent",
                        expressed=expressed,
                        is_conforming=is_conforming,
                        agent_id=agent.agent_id,
                    )
                )
                commented_ids.add(agent.agent_id)
                if expressed is Opinion.A:
                    n_a += 1
                else:
                    n_b += 1

        # Post-end updates use arrival-time support, not final post climate.
        for agent in self.agents:
            support_at_arrival = arrival_support[agent.agent_id]
            agent.apply_post_end_willingness(support_at_arrival)
            agent.record_arrival_support(support_at_arrival)
            agent.apply_post_end_confidence()
            agent.maybe_flip_opinion(self.enable_opinion_flip)

        snapshots = [
            AgentStateSnapshot(
                agent_id=agent.agent_id,
                private_opinion=agent.private_opinion,
                opinion_confidence=agent.opinion_confidence,
                willingness_to_comment=agent.willingness_to_comment,
                silence_streak=agent.silence_streak,
            )
            for agent in self.agents
        ]

        self.history.append(
            PostRecord(
                post_index=post_index,
                comments=comments,
                commented_agent_ids=commented_ids,
                arrival_support=arrival_support,
                agent_snapshots=snapshots,
            )
        )
