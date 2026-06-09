"""Agent traits, state, and update rules for the ABM simulation."""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Optional

import numpy as np


class Opinion(Enum):
    A = "A"
    B = "B"

    def opposite(self) -> Opinion:
        return Opinion.B if self is Opinion.A else Opinion.A


RECENCY_WEIGHTS = (3, 2, 1)
UPDATE_DAMPENING = 0.1


def lookback_n(stubbornness: float) -> int:
    return round((1 - stubbornness) * 3)


def weighted_climate(recent_supports: list[float], lookback: int) -> float:
    """Recency-weighted average of arrival-time support values (most recent first)."""
    if lookback <= 0 or not recent_supports:
        return 0.5
    values = recent_supports[:lookback]
    weights = np.array(RECENCY_WEIGHTS[: len(values)], dtype=float)
    weights /= weights.sum()
    return float(np.dot(weights, values))


class Agent:
    """Single agent with fixed traits and mutable state."""

    def __init__(
        self,
        agent_id: int,
        private_opinion: Opinion,
        participation_tendency: float,
        social_sensitivity: float,
        conformity_tendency: float,
        update_rate: float,
        stubbornness: float,
    ) -> None:
        self.agent_id = agent_id
        self.private_opinion = private_opinion
        self.participation_tendency = participation_tendency
        self.social_sensitivity = social_sensitivity
        self.conformity_tendency = conformity_tendency
        self.update_rate = update_rate
        self.stubbornness = stubbornness

        self.opinion_confidence = 0.5
        self.willingness_to_comment = 0.5
        self.silence_streak = 0

        # Arrival-time support_for_self per post (most recent first), max 3 posts.
        self.arrival_support_history: deque[float] = deque(maxlen=3)

    def visible_support_for_self(self, n_a: int, n_b: int) -> float:
        """Step 1: visible climate at the moment the agent arrives."""
        total = n_a + n_b
        if total == 0:
            return 0.5
        if self.private_opinion is Opinion.A:
            return n_a / total
        return n_b / total

    def p_comment(self, support_for_self: float) -> float:
        """Step 2: probability of commenting."""
        raw = (
            self.participation_tendency * 0.4
            + self.willingness_to_comment * 0.4
            + self.social_sensitivity * (support_for_self - 0.5) * 0.2
        )
        return float(np.clip(raw, 0.0, 1.0))

    def decide_to_comment(self, support_for_self: float, rng: np.random.Generator) -> bool:
        commented = bool(rng.random() < self.p_comment(support_for_self))
        if not commented:
            self.silence_streak += 1
        return commented

    def decide_expression(
        self, support_for_self: float, rng: np.random.Generator
    ) -> tuple[Opinion, bool]:
        """Step 3: return (expressed opinion, is_conforming)."""
        if support_for_self >= 0.5:
            self.silence_streak = 0
            return self.private_opinion, False

        p_conform = self.conformity_tendency * self.social_sensitivity * (1 - support_for_self)
        p_conform = float(np.clip(p_conform, 0.0, 1.0))

        if rng.random() < p_conform:
            self.silence_streak = 0
            return self.private_opinion.opposite(), True

        self.silence_streak = 0
        return self.private_opinion, False

    def apply_post_end_willingness(self, arrival_support: float) -> None:
        """Step 4: update willingness using arrival-time support_for_self."""
        climate_signal = arrival_support - 0.5
        self.willingness_to_comment += (
            self.social_sensitivity * climate_signal * UPDATE_DAMPENING
        )
        self.willingness_to_comment = float(np.clip(self.willingness_to_comment, 0.0, 1.0))

    def record_arrival_support(self, arrival_support: float) -> None:
        """Append this post's arrival-time impression to the lookback deque."""
        self.arrival_support_history.appendleft(arrival_support)

    def apply_post_end_confidence(self) -> None:
        """Step 5: update confidence using arrival-time lookback history."""
        lb = lookback_n(self.stubbornness)
        weighted = weighted_climate(list(self.arrival_support_history), lb)
        confidence_signal = weighted - 0.5
        self.opinion_confidence += (
            self.update_rate * (1 - self.stubbornness) * confidence_signal * UPDATE_DAMPENING
        )
        self.opinion_confidence = float(np.clip(self.opinion_confidence, 0.0, 1.0))

    def maybe_flip_opinion(self, enable_flip: bool) -> None:
        """Step 6: optional opinion flip when confidence collapses."""
        if not enable_flip:
            return
        if self.opinion_confidence < 0.15:
            self.private_opinion = self.private_opinion.opposite()
            self.opinion_confidence = 0.5


def create_population(
    rng: np.random.Generator,
    n_agents: int = 200,
    trait_overrides: Optional[dict[str, float]] = None,
) -> list[Agent]:
    """Create a balanced population with traits drawn from Uniform(0, 1)."""
    if n_agents == 0:
        return []

    n_a = n_agents // 2
    opinions = [Opinion.A] * n_a + [Opinion.B] * (n_agents - n_a)
    rng.shuffle(opinions)

    trait_names = (
        "participation_tendency",
        "social_sensitivity",
        "conformity_tendency",
        "update_rate",
        "stubbornness",
    )

    agents: list[Agent] = []
    for i, opinion in enumerate(opinions):
        traits = {name: float(rng.random()) for name in trait_names}
        if trait_overrides:
            traits.update(trait_overrides)
        agents.append(
            Agent(
                agent_id=i,
                private_opinion=opinion,
                participation_tendency=traits["participation_tendency"],
                social_sensitivity=traits["social_sensitivity"],
                conformity_tendency=traits["conformity_tendency"],
                update_rate=traits["update_rate"],
                stubbornness=traits["stubbornness"],
            )
        )
    return agents
