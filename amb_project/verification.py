"""Verification and ablation checks for the ABM simulation."""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np
import pandas as pd

from agents import Agent, Opinion, create_population
from environment import Condition, Simulation
from metrics import compute_run_metrics


def _run(condition: Condition, seed: int, **kwargs) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sim = Simulation(condition=condition, rng=rng, **kwargs)
    sim.run()
    return compute_run_metrics(sim)


def _mean_at_post(df: pd.DataFrame, post_index: int, column: str) -> float:
    row = df.loc[df["post_index"] == post_index]
    if row.empty:
        return 0.0
    return float(row[column].iloc[0])


def _assert_probabilities_in_range(agents: list[Agent], post_record) -> None:
    for agent in agents:
        support = post_record.arrival_support[agent.agent_id]
        p_comment = agent.p_comment(support)
        assert 0.0 <= p_comment <= 1.0, f"p_comment out of range: {p_comment}"

        if support < 0.5:
            p_conform = agent.conformity_tendency * agent.social_sensitivity * (1 - support)
            p_conform = float(np.clip(p_conform, 0.0, 1.0))
            assert 0.0 <= p_conform <= 1.0, f"p_conform out of range: {p_conform}"


def _assert_state_in_range(agents: list[Agent]) -> None:
    for agent in agents:
        assert 0.0 <= agent.opinion_confidence <= 1.0
        assert 0.0 <= agent.willingness_to_comment <= 1.0


def check_deterministic() -> None:
    """Run 1: same seed produces identical metrics."""
    df1 = _run(Condition.TREATMENT1, seed=123)
    df2 = _run(Condition.TREATMENT1, seed=123)
    pd.testing.assert_frame_equal(df1, df2)
    print("PASS: deterministic seed check")


def check_population_invariants() -> None:
    """Run 2: agent counts and bounded probabilities/states after every post."""
    rng = np.random.default_rng(7)
    sim = Simulation(condition=Condition.TREATMENT1, rng=rng)
    sim.run()

    assert len(sim.agents) == 200
    n_a = sum(1 for a in sim.agents if a.private_opinion is Opinion.A)
    n_b = sum(1 for a in sim.agents if a.private_opinion is Opinion.B)

    for post_record in sim.history:
        assert len(sim.agents) == 200
        assert n_a + n_b == 200
        _assert_probabilities_in_range(sim.agents, post_record)
        _assert_state_in_range(sim.agents)

    print("PASS: population invariant check")


def check_balanced_initialization() -> None:
    """Run 3: balanced opinions and initial state means."""
    rng = np.random.default_rng(99)
    agents = create_population(rng)

    n_a = sum(1 for a in agents if a.private_opinion is Opinion.A)
    n_b = sum(1 for a in agents if a.private_opinion is Opinion.B)
    assert n_a == 100
    assert n_b == 100

    mean_confidence = np.mean([a.opinion_confidence for a in agents])
    mean_willingness = np.mean([a.willingness_to_comment for a in agents])
    assert abs(mean_confidence - 0.5) <= 0.01
    assert abs(mean_willingness - 0.5) <= 0.01

    print("PASS: balanced initialization check")


def check_edge_cases() -> None:
    """Run 4: edge configurations complete without error."""
    # 0 agents
    rng = np.random.default_rng(1)
    sim_zero = Simulation(condition=Condition.CONTROL, rng=rng, n_agents=0)
    history = sim_zero.run()
    assert history == []
    assert compute_run_metrics(sim_zero).empty

    # 1 agent
    sim_one = Simulation(condition=Condition.CONTROL, rng=np.random.default_rng(2), n_agents=1)
    sim_one.run()
    assert len(sim_one.history) == 40

    # All stubbornness = 1.0
    df_stubborn = _run(
        Condition.TREATMENT1,
        seed=3,
        trait_overrides={"stubbornness": 1.0},
    )
    conf_start = _mean_at_post(df_stubborn, 1, "mean_confidence_B")
    conf_end = _mean_at_post(df_stubborn, 40, "mean_confidence_B")
    assert abs(conf_end - conf_start) < 0.05

    # All social_sensitivity = 0.0
    df_insensitive = _run(
        Condition.TREATMENT3,
        seed=4,
        trait_overrides={"social_sensitivity": 0.0},
    )
    gap_end = _mean_at_post(df_insensitive, 40, "participation_gap")
    assert abs(gap_end) < 0.15

    # All conformity_tendency = 0.0
    df_no_conform = _run(
        Condition.TREATMENT1,
        seed=5,
        trait_overrides={"conformity_tendency": 0.0},
    )
    assert _mean_at_post(df_no_conform, 20, "conformity_rate") < 0.05

    print("PASS: edge case checks")


def check_ablations() -> None:
    """Run 5: disabling mechanisms weakens suppression/conformity effects."""
    baseline = _run(Condition.TREATMENT3, seed=10)

    # No confidence update — confidence should stay elevated vs baseline suppression
    no_confidence = _run(
        Condition.TREATMENT3,
        seed=10,
        trait_overrides={"update_rate": 0.0},
    )
    baseline_conf_p20 = _mean_at_post(baseline, 20, "mean_confidence_B")
    ablated_conf_p20 = _mean_at_post(no_confidence, 20, "mean_confidence_B")
    assert ablated_conf_p20 > baseline_conf_p20, (
        f"no-confidence ablation should retain higher B confidence "
        f"(baseline={baseline_conf_p20:.4f}, ablated={ablated_conf_p20:.4f})"
    )

    # No willingness update (social_sensitivity=0 freezes willingness at 0.5)
    no_willingness = _run(
        Condition.TREATMENT3,
        seed=10,
        trait_overrides={"social_sensitivity": 0.0},
    )
    baseline_will_p20 = _mean_at_post(baseline, 20, "mean_willingness_B")
    ablated_will_p20 = _mean_at_post(no_willingness, 20, "mean_willingness_B")
    assert ablated_will_p20 > baseline_will_p20, (
        f"no-willingness ablation should retain higher B willingness "
        f"(baseline={baseline_will_p20:.4f}, ablated={ablated_will_p20:.4f})"
    )

    # No conformity
    no_conform = _run(
        Condition.TREATMENT1,
        seed=11,
        trait_overrides={"conformity_tendency": 0.0},
    )
    with_conform = _run(Condition.TREATMENT1, seed=11)
    assert _mean_at_post(no_conform, 20, "conformity_rate") < _mean_at_post(
        with_conform, 20, "conformity_rate"
    ), "no-conformity ablation should reduce conformity rate"

    # No lookback
    no_lookback = _run(
        Condition.TREATMENT3,
        seed=12,
        trait_overrides={"stubbornness": 1.0},
    )
    conf_change = abs(
        _mean_at_post(no_lookback, 40, "mean_confidence_B")
        - _mean_at_post(no_lookback, 1, "mean_confidence_B")
    )
    assert conf_change < 0.05, (
        f"no-lookback ablation should prevent confidence accumulation (change={conf_change:.4f})"
    )

    # No opinion flip
    rng = np.random.default_rng(13)
    agents = create_population(rng)
    initial_opinions = {a.agent_id: a.private_opinion for a in agents}
    sim = Simulation(
        condition=Condition.TREATMENT3,
        rng=np.random.default_rng(13),
        agents=agents,
        enable_opinion_flip=False,
    )
    sim.run()
    for agent in sim.agents:
        assert agent.private_opinion is initial_opinions[agent.agent_id]

    print("PASS: ablation checks")


def check_sensitivity() -> None:
    """Run 6: 20 seeds show consistent directional trends across conditions."""
    seeds = range(20)
    post_index = 40

    gaps: dict[Condition, list[float]] = {c: [] for c in Condition}
    for seed in seeds:
        for condition in Condition:
            df = _run(condition, seed=seed)
            gaps[condition].append(_mean_at_post(df, post_index, "participation_gap"))

    for condition, values in gaps.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        print(f"  {condition.value} post-{post_index} participation_gap: {mean:.4f} ± {std:.4f}")

    control_mean = float(np.mean(gaps[Condition.CONTROL]))
    t1_mean = float(np.mean(gaps[Condition.TREATMENT1]))
    t3_mean = float(np.mean(gaps[Condition.TREATMENT3]))

    assert t3_mean > control_mean, "Treatment 3 should exceed control participation gap"
    assert t3_mean > t1_mean, "Treatment 3 should exceed Treatment 1 participation gap"
    assert abs(control_mean) < 0.1, "Control should show near-zero participation gap"

    print("PASS: sensitivity check")


CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("deterministic seed", check_deterministic),
    ("population invariants", check_population_invariants),
    ("balanced initialization", check_balanced_initialization),
    ("edge cases", check_edge_cases),
    ("ablations", check_ablations),
    ("sensitivity", check_sensitivity),
]


def main() -> None:
    failures: list[str] = []
    for name, check_fn in CHECKS:
        try:
            check_fn()
        except Exception as exc:
            failures.append(f"FAIL: {name} — {exc}")

    if failures:
        print("\nVerification failed:")
        for msg in failures:
            print(f"  {msg}")
        sys.exit(1)

    print(f"\nAll {len(CHECKS)} verification checks passed.")


if __name__ == "__main__":
    main()
