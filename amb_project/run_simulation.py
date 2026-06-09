"""Entry point: run all experimental conditions and save results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from environment import Condition, Simulation
from make_figures import generate_all_figures
from metrics import aggregate_across_seeds, compute_run_metrics, save_metrics_csv

CONDITION_DIRS = {
    Condition.CONTROL: "control_neutral_all40",
    Condition.TREATMENT1: "treatment1_recovery_to_neutral",
    Condition.TREATMENT2: "treatment2_reversal_A_to_B",
    Condition.TREATMENT3: "treatment3_continued_A_to_A",
    Condition.FAVOR_B_CONTROL: "mirror_control_neutral_all40",
    Condition.FAVOR_B_TREATMENT1: "mirror_treatment1_recovery_to_neutral",
    Condition.FAVOR_B_TREATMENT2: "mirror_treatment2_reversal_B_to_A",
    Condition.FAVOR_B_TREATMENT3: "mirror_treatment3_continued_B_to_B",
}


def run_condition(condition: Condition, seed: int) -> tuple[Simulation, pd.DataFrame]:
    """Run one condition with a fresh RNG seeded for reproducible agent traits."""
    rng = np.random.default_rng(seed)
    simulation = Simulation(condition=condition, rng=rng)
    simulation.run()
    return simulation, compute_run_metrics(simulation)


def print_symmetry_check(dfs_by_condition: dict[Condition, pd.DataFrame]) -> None:
    """Compare favor-A and favor-B mirror conditions at posts 20 and 40."""
    pairs = [
        (
            Condition.TREATMENT1,
            "participation_rate_B",
            Condition.FAVOR_B_TREATMENT1,
            "participation_rate_A",
        ),
        (
            Condition.TREATMENT3,
            "participation_rate_B",
            Condition.FAVOR_B_TREATMENT3,
            "participation_rate_A",
        ),
    ]

    print("\n=== Symmetry Check (favor-A vs favor-B mirrors) ===")
    print(f"{'Post':<6} {'Original':<28} {'Mirror':<28} {'Diff':<8}")
    print("-" * 72)

    for post in (20, 40):
        for orig_cond, orig_col, mirror_cond, mirror_col in pairs:
            orig_val = float(
                dfs_by_condition[orig_cond]
                .loc[dfs_by_condition[orig_cond]["post_index"] == post, orig_col]
                .iloc[0]
            )
            mirror_val = float(
                dfs_by_condition[mirror_cond]
                .loc[dfs_by_condition[mirror_cond]["post_index"] == post, mirror_col]
                .iloc[0]
            )
            diff = orig_val - mirror_val
            label_orig = f"{orig_cond.value} {orig_col}"
            label_mirror = f"{mirror_cond.value} {mirror_col}"
            print(f"{post:<6} {orig_val:>6.4f} ({label_orig:<18}) {mirror_val:>6.4f} ({label_mirror:<18}) {diff:>+7.4f}")

    control_gap_40 = float(
        dfs_by_condition[Condition.CONTROL]
        .loc[dfs_by_condition[Condition.CONTROL]["post_index"] == 40, "participation_gap"]
        .iloc[0]
    )
    fb_control_gap_40 = float(
        dfs_by_condition[Condition.FAVOR_B_CONTROL]
        .loc[dfs_by_condition[Condition.FAVOR_B_CONTROL]["post_index"] == 40, "participation_gap"]
        .iloc[0]
    )
    print(
        f"\nControl vs mirror control participation_gap @ post 40: "
        f"{control_gap_40:.4f} vs {fb_control_gap_40:.4f} "
        f"(diff {control_gap_40 - fb_control_gap_40:+.4f})"
    )


def run_all_conditions(seed: int, output_dir: Path) -> dict[Condition, pd.DataFrame]:
    """Run all conditions and save per-condition CSVs."""
    dfs_by_condition: dict[Condition, pd.DataFrame] = {}

    for condition in Condition:
        _, df = run_condition(condition, seed)
        dfs_by_condition[condition] = df

        csv_path = output_dir / CONDITION_DIRS[condition] / "post_metrics.csv"
        save_metrics_csv(df, csv_path)
        print(f"Saved {csv_path}")

    print_symmetry_check(dfs_by_condition)

    return dfs_by_condition


def run_sensitivity(n_seeds: int, output_dir: Path) -> None:
    """Run multiple seeds per condition and save aggregated summary."""
    summaries = []

    for condition in Condition:
        dfs = []
        for seed in range(n_seeds):
            _, df = run_condition(condition, seed=seed)
            dfs.append(df)

        summary = aggregate_across_seeds(dfs)
        summary.insert(0, "condition", condition.value)
        summaries.append(summary)
        print(f"Aggregated {n_seeds} seeds for {condition.value}")

    combined = pd.concat(summaries, ignore_index=True)
    out_path = output_dir / "sensitivity_summary.csv"
    save_metrics_csv(combined, out_path)
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABM path-dependence simulation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis across multiple seeds",
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=20,
        help="Number of seeds for sensitivity mode (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running all conditions (seed={args.seed})...")
    run_all_conditions(seed=args.seed, output_dir=args.output_dir)

    if args.sensitivity:
        print(f"\nRunning sensitivity analysis ({args.sensitivity_seeds} seeds)...")
        run_sensitivity(n_seeds=args.sensitivity_seeds, output_dir=args.output_dir)

    print("\nGenerating official figures...")
    try:
        figure_paths = generate_all_figures(args.output_dir)
        print(f"Saved {len(figure_paths)} figures to {args.output_dir / 'figures'}/")
        for path in figure_paths:
            print(f"  - {path}")
    except FileNotFoundError as exc:
        print(f"Figure generation skipped: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
