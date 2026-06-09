# ABM Path Dependence — Project Skill Reference

Context-management reference for Cursor prompts working on this codebase.

## Research Question

When a balanced population is repeatedly exposed to comment sections biased toward one opinion in early visible comments, does the suppressed group's willingness to comment recover when the environment changes — or does early bias leave a lasting effect?

## Agent Parameters

### Fixed Traits (sampled once from Uniform(0, 1), never updated)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `participation_tendency` | float | [0, 1] | Baseline willingness to comment |
| `social_sensitivity` | float | [0, 1] | Strength of climate influence on behavior and belief |
| `conformity_tendency` | float | [0, 1] | Probability of expressing majority when in minority |
| `update_rate` | float | [0, 1] | Speed of confidence updates |
| `stubbornness` | float | [0, 1] | Resistance to social influence; controls lookback depth |

### Dynamic State (updated after each post)

| Variable | Type | Initial | Range | Description |
|----------|------|---------|-------|-------------|
| `private_opinion` | Opinion (A/B) | 50/50 split | A or B | True underlying opinion |
| `opinion_confidence` | float | 0.5 | [0, 1] | Belief that opinion is socially acceptable |
| `willingness_to_comment` | float | 0.5 | [0, 1] | Current motivation to participate |
| `silence_streak` | int | 0 | ≥ 0 | Consecutive posts without commenting |
| `arrival_support_history` | deque[float] | empty | maxlen 3 | Arrival-time support_for_self per post |

## Update Rules (Pseudocode)

### Step 1 — Visible climate at arrival

```
support_for_self = N_self / (N_A + N_B)   # among comments visible when agent arrives
if N_A + N_B == 0: support_for_self = 0.5
```

Seeds count in N_A/N_B. Recorded immediately in `arrival_support[agent_id]` before comment decisions.

### Step 2 — Comment decision

```
p_comment = participation_tendency*0.4 + willingness*0.4 + social_sensitivity*(support_for_self - 0.5)*0.2
p_comment = clip(p_comment, 0, 1)
if not Bernoulli(p_comment): silence_streak += 1
```

### Step 3 — Expression (if commenting)

```
if support_for_self >= 0.5:
    express private_opinion
else:
    p_conform = clip(conformity_tendency * social_sensitivity * (1 - support_for_self), 0, 1)
    if Bernoulli(p_conform): express majority opinion (conforming)
    else: express private_opinion
    silence_streak = 0
```

### Step 4 — Willingness update (post-end, uses arrival-time support)

```
# Control / favor_b_control only: damp weak signals + revert toward participation_tendency
willingness += social_sensitivity * (arrival_support - 0.5)
willingness = clip(willingness, 0, 1)
```

### Step 5 — Confidence update (post-end, arrival-time lookback)

```
lookback_n = round((1 - stubbornness) * 3)
append arrival_support to arrival_support_history (max 3)
weighted_climate = recency_weighted_average(history, weights 3/2/1 normalized)
confidence += update_rate * (1 - stubbornness) * (weighted_climate - 0.5)
confidence = clip(confidence, 0, 1)
```

**Control stabilization:** On `CONTROL` and `FAVOR_B_CONTROL` only, willingness/confidence updates apply climate damping and baseline reversion to prevent runaway feedback under neutral seeds. Treatment conditions use the original update rules even during neutral seed phases.

### Step 6 — Opinion flip (off by default)

```
if enable_opinion_flip and confidence < 0.15:
    flip private_opinion; confidence = 0.5
```

## Experimental Conditions

| Condition | Posts 1–20 | Posts 21–40 |
|-----------|-----------|-------------|
| Control | Neutral seeds | Neutral seeds |
| Treatment 1 | Favor-A seeds | Neutral seeds |
| Treatment 2 | Favor-A seeds | Favor-B seeds |
| Treatment 3 | Favor-A seeds | Favor-A seeds |
| Favor-B Control | Neutral seeds | Neutral seeds |
| Favor-B Treatment 1 | Favor-B seeds | Neutral seeds |
| Favor-B Treatment 2 | Favor-B seeds | Favor-A seeds |
| Favor-B Treatment 3 | Favor-B seeds | Favor-B seeds |

## File Structure

| File | Responsibility |
|------|----------------|
| `agents.py` | Agent class, traits, state, update rules, `create_population()` |
| `environment.py` | Conditions, seed generation, arrival queue, `Simulation` orchestration |
| `metrics.py` | Outcome measures and CSV export |
| `make_figures.py` | Official publication figures via `generate_all_figures()` |
| `verification.py` | Six verification/ablation check categories |
| `run_simulation.py` | CLI entry point; runs conditions; calls `generate_all_figures()` |
| `PROMPT.md` | Full project specification |
| `PLAN.md` | Implementation plan with design decisions |

## Outcome Measures

Per post, per condition. Key note: **`comment_imbalance` uses agent-generated comments only** (seeds excluded). `conformity_rate` also excludes seeds.

- `comment_imbalance`, `participation_rate_A/B`, `participation_gap`
- `mean_confidence_A/B`, `mean_willingness_A/B`
- `conformity_rate`, silence streak summaries
- `cascade_flag` — imbalance > 0.3 threshold

## Verification Checks (`python verification.py`)

| Check | Tests |
|-------|-------|
| 1. Deterministic seed | Same seed → identical metrics |
| 2. Population invariants | 200 agents; bounded probabilities and state |
| 3. Balanced initialization | 100 A / 100 B; mean confidence & willingness = 0.5 |
| 4. Edge cases | 0 agents, 1 agent, stubbornness=1, sensitivity=0, conformity=0 |
| 5. Ablations | Disabling each mechanism weakens suppression effects |
| 6. Sensitivity | 20 seeds; consistent directional trends across conditions |

## Common Failure Modes

1. **Confidence/willingness updated before arrival queue completes** — agents would use incomplete climate signals; post-end updates must run only after all agents have arrived.

2. **Using post-end climate instead of arrival-time support** — willingness and confidence lookback must use each agent's `support_for_self` at their arrival moment, not the final comment-section ratio.

3. **Seed comments excluded from climate calculation** — agents arriving first would see 0 comments (support=0.5) instead of biased seeds; suppression would not initialize correctly. Seeds must be in `n_a`/`n_b` before the first agent arrives.

4. **Seed comments included in `comment_imbalance`** — metric would reflect seed setup, not agent behavior. Agent-only comments must be used for imbalance and cascade detection.

5. **Control condition receives favor-A seeds** — baseline would show artificial suppression; Control must be neutral for all 40 posts.

## Coding Conventions

- All probabilities clipped with `np.clip(..., 0, 1)`
- All random draws via a single seeded `numpy.random.Generator` passed at initialization
- No global mutable state outside `Simulation` (in `environment.py`)
- Results saved as CSV in `results/<condition_folder>/`; figures as PNG in `results/figures/`
- Output folders: `control_neutral_all40`, `treatment1_recovery_to_neutral`, `treatment2_reversal_A_to_B`, `treatment3_continued_A_to_A`, and `mirror_*` counterparts
- Official figures: `fig1_participation_gap.png` through `fig6_mirror_symmetry.png`
