# ABM: Path Dependence in Online Opinion Climates

A Python agent-based model studying whether early visible comment bias suppresses minority participation — and whether that suppression persists after the environment changes.

## Research Question

When a balanced population is repeatedly exposed to comment sections favoring one opinion in early visible comments, does the suppressed group's willingness to comment recover when conditions change — or does early bias create a lasting effect?

See [PROMPT.md](PROMPT.md) for the full specification.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

### Run simulation and generate figures

```bash
python run_simulation.py --sensitivity --sensitivity-seeds 20
```

This runs all eight conditions, aggregates results across seeds, and generates the official figures via `make_figures.py`.

Options:

```bash
python run_simulation.py --seed 42 --output-dir results
python run_simulation.py --sensitivity --sensitivity-seeds 20
```

### Generate figures only (from existing results)

If `results/sensitivity_summary.csv` already exists:

```bash
python make_figures.py
```

### Run verification checks

```bash
python verification.py
```

## Experimental Conditions

| Condition | Folder | Phase 1 (posts 1–20) | Phase 2 (posts 21–40) |
|-----------|--------|------------------------|------------------------|
| Control | `control_neutral_all40` | Neutral | Neutral |
| Treatment 1 — Recovery | `treatment1_recovery_to_neutral` | Favor-A | Neutral |
| Treatment 2 — Reversal | `treatment2_reversal_A_to_B` | Favor-A | Favor-B |
| Treatment 3 — Continued | `treatment3_continued_A_to_A` | Favor-A | Favor-A |

Mirror (favor-B) conditions use the `mirror_*` folders and follow the same phase structure with A and B reversed.

## Output Structure

```
results/
├── control_neutral_all40/post_metrics.csv
├── treatment1_recovery_to_neutral/post_metrics.csv
├── treatment2_reversal_A_to_B/post_metrics.csv
├── treatment3_continued_A_to_A/post_metrics.csv
├── mirror_control_neutral_all40/post_metrics.csv
├── mirror_treatment1_recovery_to_neutral/post_metrics.csv
├── mirror_treatment2_reversal_B_to_A/post_metrics.csv
├── mirror_treatment3_continued_B_to_B/post_metrics.csv
├── sensitivity_summary.csv              # with --sensitivity
└── figures/
    ├── fig1_participation_gap.png
    ├── fig2_willingness_gap.png
    ├── fig3_confidence_gap.png
    ├── fig4_final_participation_gap.png
    ├── fig5_recovery_comparison.png
    └── fig6_mirror_symmetry.png
```

`comment_imbalance` measures agent-generated comments only (seeds excluded).

## Project Structure

| File | Purpose |
|------|---------|
| `agents.py` | Agent traits, state, update rules |
| `environment.py` | Simulation loop, seeds, arrival queue |
| `metrics.py` | Outcome measures and CSV export |
| `verification.py` | Verification and ablation checks |
| `run_simulation.py` | CLI entry point; runs conditions and calls `generate_all_figures()` |
| `make_figures.py` | Official publication figure generation (`generate_all_figures`) |
| `SKILL.md` | Cursor context reference |
| `PLAN.md` | Implementation plan |
| `PROMPT.md` | Full project specification |

## Docker

```bash
docker build -t abm-simulation .
docker run --rm -v "$(pwd)/results:/app/results" abm-simulation
```

Mount `results/` to persist output outside the container.

## Results

Figures are generated from `sensitivity_summary.csv` (20-seed means and standard deviations). Run with `--sensitivity` to produce this file before figure generation.

### Figure 1 — Participation Gap Over Time

![Figure 1: Participation Gap Over Time](results/figures/fig1_participation_gap.png)

**Caption:** Participation gap (rate A − rate B) across 40 posts for Control, Recovery, Reversal, and Continued conditions.

**Interpretation:** All treatment conditions share identical Phase 1 favor-A seeds, producing a widening participation gap through post 20. After the environment changes at post 21, Recovery shows persistent elevation, Reversal trends back toward zero, and Continued deepens the gap. Control remains near zero throughout.

### Figure 2 — Willingness Gap Over Time

![Figure 2: Willingness Gap Over Time](results/figures/fig2_willingness_gap.png)

**Caption:** Mean willingness gap (A − B) across posts for the four main conditions.

**Interpretation:** Willingness diverges in parallel with participation under biased seeds, reflecting the post-end willingness update pathway. The gap persists or recovers depending on Phase 2 seed bias, providing one mechanism underlying participation differences.

### Figure 3 — Confidence Gap Over Time

![Figure 3: Confidence Gap Over Time](results/figures/fig3_confidence_gap.png)

**Caption:** Mean confidence gap (A − B) across posts for the four main conditions.

**Interpretation:** Confidence shifts more gradually than willingness, consistent with the lookback-weighted update rule. Treatment conditions show growing separation in perceived social acceptability, while Control stays near zero.

### Figure 4 — Final Participation Gap Summary

![Figure 4: Final Participation Gap Summary](results/figures/fig4_final_participation_gap.png)

**Caption:** Participation gap at post 40 with error bars (standard deviation across 20 seeds).

**Interpretation:** Continued suppression produces the largest final gap; Recovery maintains a substantial but smaller gap; Reversal returns near baseline; Control stays close to zero. Error bars indicate effects are robust across seeds.

### Figure 5 — Recovery vs Reversal Comparison

![Figure 5: Recovery vs Reversal Comparison](results/figures/fig5_recovery_comparison.png)

**Caption:** Participation gap over time for Recovery (A→Neutral) and Reversal (A→B) only.

**Interpretation:** After identical Phase 1 suppression, neutral Phase 2 allows partial persistence of the gap, while favor-B Phase 2 drives recovery toward balance. This comparison directly addresses whether suppression reverses when the environment changes.

### Figure 6 — Mirror Symmetry Verification

![Figure 6: Mirror Symmetry Verification](results/figures/fig6_mirror_symmetry.png)

**Caption:** Participation gap for favor-A conditions (solid) and favor-B mirror conditions (dashed).

**Interpretation:** Mirror pairs produce approximately opposite-sign gaps, confirming the model does not inherently favor Opinion A or B. Small deviations reflect independent arrival-order shuffles per condition, not structural asymmetry in the agent rules.
