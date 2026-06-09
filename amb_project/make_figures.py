"""Generate publication-style figures from existing simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DPI = 300
PHASE_TRANSITION_POST = 20

# Condition keys in sensitivity_summary.csv (unchanged — simulation identifiers)
MAIN_CONDITIONS = ["control", "treatment1", "treatment2", "treatment3"]
MIRROR_PAIRS = [
    ("treatment1", "favor_b_treatment1"),
    ("treatment2", "favor_b_treatment2"),
    ("treatment3", "favor_b_treatment3"),
]

# Display names for plotting only
CONDITION_LABELS = {
    "control": "Control",
    "treatment1": "Recovery (A→Neutral)",
    "treatment2": "Reversal (A→B)",
    "treatment3": "Continued (A→A)",
    "favor_b_treatment1": "Mirror Recovery (B→Neutral)",
    "favor_b_treatment2": "Mirror Reversal (B→A)",
    "favor_b_treatment3": "Mirror Continued (B→B)",
}

MAIN_COLORS = {
    "control": "#4C72B0",
    "treatment1": "#DD8452",
    "treatment2": "#55A868",
    "treatment3": "#C44E52",
}

MIRROR_COLORS = {
    "treatment1": "#DD8452",
    "favor_b_treatment1": "#DD8452",
    "treatment2": "#55A868",
    "favor_b_treatment2": "#55A868",
    "treatment3": "#C44E52",
    "favor_b_treatment3": "#C44E52",
}

OFFICIAL_FIGURES = [
    "fig1_participation_gap.png",
    "fig2_willingness_gap.png",
    "fig3_confidence_gap.png",
    "fig4_final_participation_gap.png",
    "fig5_recovery_comparison.png",
    "fig6_mirror_symmetry.png",
]


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_sensitivity(output_dir: Path) -> pd.DataFrame:
    sensitivity_path = output_dir / "sensitivity_summary.csv"
    if not sensitivity_path.exists():
        raise FileNotFoundError(
            f"Missing required file: {sensitivity_path}. "
            "Run with --sensitivity to generate it, or place an existing "
            "sensitivity_summary.csv in the output directory."
        )
    return pd.read_csv(sensitivity_path)


def _subset(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    return df[df["condition"] == condition].sort_values("post_index")


def _add_reference_lines(ax: plt.Axes, *, show_zero: bool = True) -> None:
    if show_zero:
        ax.axhline(0, color="#888888", linestyle="--", linewidth=1.0, zorder=0)
    ax.axvline(
        PHASE_TRANSITION_POST,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
        label="Phase 1 → Phase 2",
    )


def _save_figure(fig: plt.Figure, figures_dir: Path, filename: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure1_participation_gap(df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _add_reference_lines(ax)

    for condition in MAIN_CONDITIONS:
        sub = _subset(df, condition)
        ax.plot(
            sub["post_index"],
            sub["participation_gap_mean"],
            label=CONDITION_LABELS[condition],
            color=MAIN_COLORS[condition],
            linewidth=2.2,
        )

    ax.set_xlabel("Post")
    ax.set_ylabel("Participation Gap (A − B)")
    ax.set_title("Participation Gap Over Time")
    ax.set_xlim(1, 40)
    ax.set_xticks(range(1, 41, 5))
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "fig1_participation_gap.png")


def figure2_willingness_gap(df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _add_reference_lines(ax)

    for condition in MAIN_CONDITIONS:
        sub = _subset(df, condition)
        gap = sub["mean_willingness_A_mean"] - sub["mean_willingness_B_mean"]
        ax.plot(
            sub["post_index"],
            gap,
            label=CONDITION_LABELS[condition],
            color=MAIN_COLORS[condition],
            linewidth=2.2,
        )

    ax.set_xlabel("Post")
    ax.set_ylabel("Willingness Gap (A − B)")
    ax.set_title("Willingness Gap Over Time")
    ax.set_xlim(1, 40)
    ax.set_xticks(range(1, 41, 5))
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "fig2_willingness_gap.png")


def figure3_confidence_gap(df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _add_reference_lines(ax)

    for condition in MAIN_CONDITIONS:
        sub = _subset(df, condition)
        gap = sub["mean_confidence_A_mean"] - sub["mean_confidence_B_mean"]
        ax.plot(
            sub["post_index"],
            gap,
            label=CONDITION_LABELS[condition],
            color=MAIN_COLORS[condition],
            linewidth=2.2,
        )

    ax.set_xlabel("Post")
    ax.set_ylabel("Confidence Gap (A − B)")
    ax.set_title("Confidence Gap Over Time")
    ax.set_xlim(1, 40)
    ax.set_xticks(range(1, 41, 5))
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "fig3_confidence_gap.png")


def figure4_final_participation_gap(df: pd.DataFrame, figures_dir: Path) -> Path:
    post40 = df[df["post_index"] == 40]
    labels = [CONDITION_LABELS[c] for c in MAIN_CONDITIONS]
    means = [post40.loc[post40["condition"] == c, "participation_gap_mean"].iloc[0] for c in MAIN_CONDITIONS]
    stds = [post40.loc[post40["condition"] == c, "participation_gap_std"].iloc[0] for c in MAIN_CONDITIONS]
    colors = [MAIN_COLORS[c] for c in MAIN_CONDITIONS]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(MAIN_CONDITIONS))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )

    ax.axhline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Participation Gap (A − B)")
    ax.set_title("Final Participation Gap at Post 40")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.012 if mean >= 0 else -0.028),
            f"{mean:+.3f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    return _save_figure(fig, figures_dir, "fig4_final_participation_gap.png")


def figure5_recovery_comparison(df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvline(PHASE_TRANSITION_POST, color="#888888", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1.0)

    for condition in ["treatment1", "treatment2"]:
        sub = _subset(df, condition)
        ax.plot(
            sub["post_index"],
            sub["participation_gap_mean"],
            label=CONDITION_LABELS[condition],
            color=MAIN_COLORS[condition],
            linewidth=2.5,
        )

    ax.set_xlabel("Post")
    ax.set_ylabel("Participation Gap (A − B)")
    ax.set_title("Recovery Comparison: After Phase 1 Suppression", fontsize=16, fontweight="bold")
    ax.set_xlim(1, 40)
    ax.set_xticks(range(1, 41, 5))
    ax.legend(frameon=True, loc="best")

    ymin, ymax = ax.get_ylim()
    ax.annotate(
        "Environment changes",
        xy=(PHASE_TRANSITION_POST, ymax * 0.88),
        xytext=(PHASE_TRANSITION_POST + 5, ymax * 0.88),
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        va="center",
    )

    fig.tight_layout()
    return _save_figure(fig, figures_dir, "fig5_recovery_comparison.png")


def figure6_mirror_symmetry(df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(PHASE_TRANSITION_POST, color="#888888", linestyle="--", linewidth=1.0, label="Phase 1 → Phase 2")
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1.0)

    for favor_a, favor_b in MIRROR_PAIRS:
        color = MIRROR_COLORS[favor_a]
        sub_a = _subset(df, favor_a)
        sub_b = _subset(df, favor_b)

        ax.plot(
            sub_a["post_index"],
            sub_a["participation_gap_mean"],
            label=CONDITION_LABELS[favor_a],
            color=color,
            linewidth=2.2,
            linestyle="-",
        )
        ax.plot(
            sub_b["post_index"],
            sub_b["participation_gap_mean"],
            label=CONDITION_LABELS[favor_b],
            color=color,
            linewidth=2.2,
            linestyle="--",
        )

    ax.set_xlabel("Post")
    ax.set_ylabel("Participation Gap")
    ax.set_title("Mirror Symmetry Verification (Favor-A vs Favor-B)")
    ax.set_xlim(1, 40)
    ax.set_xticks(range(1, 41, 5))
    ax.legend(frameon=True, loc="best", ncol=2)
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "fig6_mirror_symmetry.png")


def generate_all_figures(output_dir: str | Path = "results") -> list[Path]:
    """Generate all official publication figures from sensitivity_summary.csv."""
    _apply_style()
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    df = _load_sensitivity(output_dir)

    return [
        figure1_participation_gap(df, figures_dir),
        figure2_willingness_gap(df, figures_dir),
        figure3_confidence_gap(df, figures_dir),
        figure4_final_participation_gap(df, figures_dir),
        figure5_recovery_comparison(df, figures_dir),
        figure6_mirror_symmetry(df, figures_dir),
    ]


def main() -> None:
    output_dir = Path("results")
    generated = generate_all_figures(output_dir)

    print("=" * 60)
    print("ABM Figure Generation Summary")
    print("=" * 60)
    print("\nFiles read:")
    print(f"  - {(output_dir / 'sensitivity_summary.csv').resolve()}")
    print("\nFigures generated:")
    for path in generated:
        print(f"  - {path.resolve()}")
    print(f"\nOutput directory: {(output_dir / 'figures').resolve()}")
    print(f"Total figures: {len(generated)}")


if __name__ == "__main__":
    main()
