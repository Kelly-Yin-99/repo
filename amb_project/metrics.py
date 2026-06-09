"""Outcome measure calculations, aggregation, and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import Agent, Opinion
from environment import AgentStateSnapshot, Comment, Condition, PostRecord, Simulation

CASCADE_THRESHOLD = 0.3


def _agents_by_opinion(agents: list[Agent], opinion: Opinion) -> list[Agent]:
    return [a for a in agents if a.private_opinion is opinion]


def _snapshots_by_opinion(
    snapshots: list[AgentStateSnapshot], opinion: Opinion
) -> list[AgentStateSnapshot]:
    return [s for s in snapshots if s.private_opinion is opinion]


def compute_post_metrics(agents: list[Agent], post_record: PostRecord) -> dict:
    """Compute all per-post outcome measures."""
    agent_comments = [c for c in post_record.comments if c.source == "agent"]

    # Agent-generated comments only — excludes seeds so the metric reflects agent behavior.
    if agent_comments:
        prop_a = sum(1 for c in agent_comments if c.expressed is Opinion.A) / len(agent_comments)
        prop_b = 1.0 - prop_a
        comment_imbalance = abs(prop_a - prop_b)
    else:
        comment_imbalance = 0.0

    if post_record.agent_snapshots:
        snapshots_a = _snapshots_by_opinion(post_record.agent_snapshots, Opinion.A)
        snapshots_b = _snapshots_by_opinion(post_record.agent_snapshots, Opinion.B)

        def participation_rate_snapshots(group: list[AgentStateSnapshot]) -> float:
            if not group:
                return 0.0
            commented = sum(1 for s in group if s.agent_id in post_record.commented_agent_ids)
            return commented / len(group)

        participation_rate_a = participation_rate_snapshots(snapshots_a)
        participation_rate_b = participation_rate_snapshots(snapshots_b)

        def mean_snapshot(group: list[AgentStateSnapshot], attr: str) -> float:
            if not group:
                return 0.0
            return float(np.mean([getattr(s, attr) for s in group]))

        mean_confidence_a = mean_snapshot(snapshots_a, "opinion_confidence")
        mean_confidence_b = mean_snapshot(snapshots_b, "opinion_confidence")
        mean_willingness_a = mean_snapshot(snapshots_a, "willingness_to_comment")
        mean_willingness_b = mean_snapshot(snapshots_b, "willingness_to_comment")
        silence_streaks = [s.silence_streak for s in post_record.agent_snapshots]
    else:
        agents_a = _agents_by_opinion(agents, Opinion.A)
        agents_b = _agents_by_opinion(agents, Opinion.B)

        def participation_rate(group: list[Agent]) -> float:
            if not group:
                return 0.0
            commented = sum(1 for a in group if a.agent_id in post_record.commented_agent_ids)
            return commented / len(group)

        participation_rate_a = participation_rate(agents_a)
        participation_rate_b = participation_rate(agents_b)

        def mean_state(group: list[Agent], attr: str) -> float:
            if not group:
                return 0.0
            return float(np.mean([getattr(a, attr) for a in group]))

        mean_confidence_a = mean_state(agents_a, "opinion_confidence")
        mean_confidence_b = mean_state(agents_b, "opinion_confidence")
        mean_willingness_a = mean_state(agents_a, "willingness_to_comment")
        mean_willingness_b = mean_state(agents_b, "willingness_to_comment")
        silence_streaks = [a.silence_streak for a in agents]

    if agent_comments:
        conformity_rate = sum(1 for c in agent_comments if c.is_conforming) / len(agent_comments)
    else:
        conformity_rate = 0.0

    silence_mean = float(np.mean(silence_streaks)) if silence_streaks else 0.0
    silence_max = max(silence_streaks) if silence_streaks else 0
    silence_pct_ge_5 = (
        sum(1 for s in silence_streaks if s >= 5) / len(silence_streaks) if silence_streaks else 0.0
    )

    return {
        "post_index": post_record.post_index,
        "comment_imbalance": comment_imbalance,
        "participation_rate_A": participation_rate_a,
        "participation_rate_B": participation_rate_b,
        "participation_gap": participation_rate_a - participation_rate_b,
        "mean_confidence_A": mean_confidence_a,
        "mean_confidence_B": mean_confidence_b,
        "mean_willingness_A": mean_willingness_a,
        "mean_willingness_B": mean_willingness_b,
        "conformity_rate": conformity_rate,
        "silence_streak_mean": silence_mean,
        "silence_streak_max": silence_max,
        "silence_streak_pct_ge_5": silence_pct_ge_5,
        "cascade_flag": int(comment_imbalance > CASCADE_THRESHOLD),
    }


def compute_run_metrics(simulation: Simulation) -> pd.DataFrame:
    """Build a DataFrame with one row per post for a completed simulation."""
    if not simulation.history:
        return pd.DataFrame()

    rows = [
        compute_post_metrics(simulation.agents, post_record)
        for post_record in simulation.history
    ]
    return pd.DataFrame(rows)


def cascade_flag(post_metrics: dict) -> bool:
    return post_metrics["comment_imbalance"] > CASCADE_THRESHOLD


def aggregate_across_seeds(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute mean, std, and cascade_probability per post across multiple runs."""
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, keys=range(len(dfs)), names=["run", "row"]).reset_index(level=0)
    numeric_cols = [
        c
        for c in dfs[0].columns
        if c != "post_index" and pd.api.types.is_numeric_dtype(dfs[0][c])
    ]

    grouped = combined.groupby("post_index")
    mean_df = grouped[numeric_cols].mean().add_suffix("_mean")
    std_df = grouped[numeric_cols].std().add_suffix("_std")
    cascade_prob = grouped["cascade_flag"].mean().rename("cascade_probability")

    result = pd.concat([mean_df, std_df, cascade_prob], axis=1).reset_index()
    return result


def save_metrics_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_trajectories(
    dfs_by_condition: dict[Condition, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save trajectory PNGs comparing conditions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_labels = {c: c.value for c in dfs_by_condition}
    posts = range(1, 41)

    plot_specs = [
        (
            "participation_gap",
            "Participation Gap (A - B)",
            "participation_gap.png",
        ),
        (
            ("mean_confidence_A", "mean_confidence_B"),
            "Mean Opinion Confidence",
            "mean_confidence.png",
        ),
        (
            ("mean_willingness_A", "mean_willingness_B"),
            "Mean Willingness to Comment",
            "mean_willingness.png",
        ),
        (
            "comment_imbalance",
            "Comment Imbalance",
            "comment_imbalance.png",
        ),
    ]

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(spec[0], tuple):
            for col, suffix in zip(spec[0], ("A", "B")):
                for condition, df in dfs_by_condition.items():
                    if df.empty:
                        continue
                    ax.plot(
                        df["post_index"],
                        df[col],
                        label=f"{condition_labels[condition]} ({suffix})",
                        alpha=0.8,
                    )
        else:
            col = spec[0]
            for condition, df in dfs_by_condition.items():
                if df.empty:
                    continue
                ax.plot(
                    df["post_index"],
                    df[col],
                    label=condition_labels[condition],
                    alpha=0.8,
                )

        ax.set_xlabel("Post")
        ax.set_ylabel(spec[1])
        ax.set_title(spec[1])
        ax.set_xticks(list(posts)[::4])
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / spec[2], dpi=150)
        plt.close(fig)
