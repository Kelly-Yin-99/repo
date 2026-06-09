# PROMPT.md — ABM Project: Path Dependence in Online Opinion Climates

## Project Request

I want to build a Python agent-based model about how early visible comments in online comment sections shape public opinion climates over time. Specifically, I am interested in whether repeated exposure to biased comment environments suppresses the minority-leaning group's willingness to comment, and whether that suppression persists even after the environment changes. Please read all sections below carefully before writing any code or making any design decisions.

---

## Research Question

When a balanced population of agents is repeatedly exposed to comment sections that favor one opinion in early visible comments, does the suppressed group's willingness to comment recover when the environment later changes — or does the early bias create a lasting effect that persists regardless of what comes after?

A secondary question is whether suppression and recovery are symmetric: does recovery take longer or remain incomplete compared to how quickly suppression originally developed?

These are open empirical questions in this model. The goal is to observe what the simulation produces, not to confirm a predetermined answer.

---

## Model Environment

Keep the environment minimal and interpretable.

- 200 agents with balanced private opinions (100 Opinion A, 100 Opinion B)
- 40 sequential posts on the same general topic
- Each post has a fresh comment section, but agents carry their internal states forward across posts
- Agents arrive in a **random order** within each post (reshuffled every post)
- Each agent sees only comments made before them in that post's arrival queue
- No follower networks, no likes, no recommendation systems, no natural language generation
- Comments are labels only: A or B

---

## Experimental Design

### Phase 1 (Posts 1–20): All conditions share the same environment
All four conditions run through 20 posts with **favor-A seed comments** (approximately 70% A, 30% B in seed comments). This establishes suppression of B-opinion agents uniformly across conditions before they diverge.

### Phase 2 (Posts 21–40): Conditions diverge

| Condition | Phase 2 Environment | Purpose |
|-----------|-------------------|---------|
| Control | Neutral (50/50 seed) for all 40 posts | Baseline — no suppression phase at all |
| Treatment 1 | Neutral seed comments | Does suppression persist without continued pressure? |
| Treatment 2 | Favor-B seed comments (~70% B) | Can a reversed environment undo suppression? |
| Treatment 3 | Favor-A seed comments (continued) | Does suppression deepen with continued pressure? |

The control condition runs neutral posts for all 40 posts, providing a baseline participation rate for both A and B agents in the absence of any bias. Treatments 1–3 all share the same Phase 1 and diverge only in Phase 2.

---

## Agent Properties

### Fixed Traits — sampled once at initialization, never updated
All traits drawn from **Uniform(0, 1)** and kept fixed throughout the simulation.

| Parameter | Description |
|-----------|-------------|
| `participation_tendency` | Baseline willingness to comment, independent of environment |
| `social_sensitivity` | How strongly visible comment climate influences behavior and belief updating |
| `conformity_tendency` | Probability of expressing majority opinion when private opinion conflicts with visible climate |
| `update_rate` | Speed at which confidence updates in response to climate signals |
| `stubbornness` | Resistance to social influence; also determines how many recent posts the agent looks back on |

### Dynamic State Variables — updated after each post
| Variable | Initial Value | Description |
|----------|--------------|-------------|
| `private_opinion` | A or B (50/50) | The agent's true underlying opinion; fixed unless confidence collapses |
| `opinion_confidence` | 0.5 | How strongly the agent believes their opinion is socially acceptable; carries across posts |
| `willingness_to_comment` | 0.5 | Current motivation to participate; updated by climate each post |
| `silence_streak` | 0 | Number of consecutive posts the agent has not commented |

---

## Memory and Lookback Mechanism

After each post, agents update their internal states by reviewing recent post history. The number of posts they look back on depends on their stubbornness — stubborn agents do not bother looking back at what others said, while less stubborn agents are more attentive to recent social history:

```
lookback_n = round((1 - stubbornness) * 3)
# stubbornness = 0.9 → looks back 0 posts (ignores history)
# stubbornness = 0.5 → looks back 2 posts
# stubbornness = 0.1 → looks back 3 posts
```

When computing the weighted climate from recent posts, apply recency weighting:
- Most recent post: weight 3
- Two posts ago: weight 2
- Three posts ago: weight 1
- Normalize weights to sum to 1

If fewer than lookback_n posts are available (early in the simulation), use however many exist.

---

## Update Rules

### Step 1: Compute visible climate (within each post, as agent arrives)

```
support_A = N_A / (N_A + N_B)    # among comments visible so far this post
support_B = N_B / (N_A + N_B)

if private_opinion == A:
    support_for_self = support_A
else:
    support_for_self = support_B

# Edge case: if no comments visible yet
if N_A + N_B == 0:
    support_for_self = 0.5
```

Seed comments are present before any agent arrives and count identically to regular comments in the climate calculation.

### Step 2: Decide whether to comment

```
p_comment = (participation_tendency * 0.4
           + willingness_to_comment * 0.4
           + social_sensitivity * (support_for_self - 0.5) * 0.2)

p_comment = clip(p_comment, 0, 1)
```

Draw from Bernoulli(p_comment). If the agent does not comment, increment silence_streak.

### Step 3: Decide what to express (only if commenting)

```
if support_for_self >= 0.5:
    # Agent's opinion matches or leads the visible climate
    express private_opinion

else:
    # Agent's opinion is in the minority
    p_conform = conformity_tendency * social_sensitivity * (1 - support_for_self)
    p_conform = clip(p_conform, 0, 1)

    if Bernoulli(p_conform):
        express majority opinion
    else:
        express private_opinion
```

If the agent expresses a comment (whether private or conformed), reset silence_streak to 0 and add the comment to the visible section for agents arriving later in this post.

### Step 4: Update willingness_to_comment (after each post ends)

```
climate_signal = support_for_self - 0.5

willingness_to_comment += social_sensitivity * climate_signal
willingness_to_comment = clip(willingness_to_comment, 0, 1)
```

### Step 5: Update opinion_confidence (after each post, using lookback)

```
lookback_n = round((1 - stubbornness) * 3)

# Gather weighted climate from recent posts
weighted_climate = recency_weighted_average(recent_posts, lookback_n)
# weighted_climate is support_for_self averaged across recent posts with recency weights

confidence_signal = weighted_climate - 0.5
opinion_confidence += update_rate * (1 - stubbornness) * confidence_signal
opinion_confidence = clip(opinion_confidence, 0, 1)
```

### Step 6: Check for private opinion flip (optional mechanism, off by default)

```
if opinion_confidence < 0.15:
    private_opinion = flip to opposite
    opinion_confidence = 0.5   # reset after flip
```

This represents agents who have been so consistently in the minority that they genuinely begin to doubt their own view. Include as a toggleable mechanism for ablation testing, but keep it off by default.

---

## Seed Comment Generation

Before each post begins, generate approximately 20 seed comments. Seed composition depends on the experimental condition:

- **Neutral**: ~50% A, ~50% B
- **Favor-A**: ~70% A, ~30% B
- **Favor-B**: ~70% B, ~30% A

Seed comments are visible to all arriving agents and function identically to regular comments in the climate calculation.

---

## Outcome Measures

Record the following after each post, separately for each condition:

| Measure | Definition |
|---------|------------|
| `comment_imbalance` | \|proportion_A - proportion_B\| among all comments in that post |
| `participation_rate_A` | Proportion of A-opinion agents who commented this post |
| `participation_rate_B` | Proportion of B-opinion agents who commented this post |
| `participation_gap` | participation_rate_A - participation_rate_B |
| `mean_confidence_A` | Mean opinion_confidence across all A agents |
| `mean_confidence_B` | Mean opinion_confidence across all B agents |
| `mean_willingness_A` | Mean willingness_to_comment across all A agents |
| `mean_willingness_B` | Mean willingness_to_comment across all B agents |
| `conformity_rate` | Proportion of comments that were conforming expressions |
| `silence_streak_dist` | Distribution of silence streaks across all agents |
| `cascade_probability` | Proportion of simulation runs where imbalance exceeds 0.3 threshold |

Track all measures across all 40 posts so trajectory plots are possible.

---

## Project File Structure

```
abm-project/
├── run_simulation.py       # Entry point; runs all conditions and saves results
├── agents.py               # Agent class with all traits, states, and update rules
├── environment.py          # Post/comment section logic; seed generation; arrival queue
├── metrics.py              # All outcome measure calculations
├── verification.py         # All verification and ablation checks
├── SKILL.md                # Context-management file
├── results/                # Auto-generated output folder
│   ├── condition_control/
│   ├── condition_treatment1/
│   ├── condition_treatment2/
│   ├── condition_treatment3/
│   └── figures/
├── PROMPT.md
├── PLAN.md
├── README.md
└── Dockerfile
```

---

## Verification Requirements

All checks must be runnable via `python verification.py`.

### 1. Deterministic Seed Check
Run the full simulation twice with the same random seed. Assert that all outcome measures are identical across both runs.

### 2. Population Invariant Check
After every post, assert:
- Total number of agents = 200
- Number of A-opinion agents + B-opinion agents = 200
- All probabilities (p_comment, p_conform) remain in [0, 1]
- All state variables (confidence, willingness) remain in [0, 1]

### 3. Balanced Initialization Check
After initialization, assert:
- Exactly 100 agents hold Opinion A and 100 hold Opinion B
- Mean confidence = 0.5 ± 0.01
- Mean willingness = 0.5 ± 0.01

### 4. Edge Case Checks
Run the model under:
- 0 agents (should not crash)
- 1 agent (should run without errors)
- All agents have stubbornness = 1.0 (no lookback, no confidence update)
- All agents have social_sensitivity = 0.0 (no climate influence — participation should be near-uniform with no suppression)
- All agents have conformity_tendency = 0.0 (no conformity — only silence mechanism remains)

### 5. Ablation Checks
Each ablation removes one mechanism and verifies the effect weakens:

| Ablation | What to disable | Expected result |
|----------|----------------|-----------------|
| No confidence update | Set update_rate = 0 for all agents | Suppression should appear only via willingness, not deepen across posts |
| No willingness update | Hold willingness fixed at 0.5 | Suppression should appear only via confidence pathway |
| No conformity | Set conformity_tendency = 0 | Comment content should reflect true private opinions more |
| No lookback | Set stubbornness = 1.0 for all | Confidence should not accumulate across posts |
| No opinion flip | Disable flip mechanism | No agents should change private opinion |

### 6. Sensitivity Check
Run each condition with 20 different random seeds. Report mean and standard deviation of key outcome measures across seeds. Results should show consistent directional trends, not just single-run artifacts.

---

## SKILL.md Requirements

The SKILL.md file must encode project-specific knowledge for context management across Cursor prompts. Include:

- Summary of all agent parameters and state variables with types and ranges
- All update rules in pseudocode
- File structure and what each file is responsible for
- List of all verification checks and what each one tests
- Common failure modes: what goes wrong if confidence update is applied before arrival order is processed; what happens if seed comments are excluded from the climate calculation
- Coding conventions: all probabilities clipped with np.clip; all random draws via a single seeded numpy RNG instance passed at initialization; no global state outside the Environment class

---

## Constraints and Simplifications

- No natural language generation
- No likes, upvotes, or follower networks
- No recommendation or ranking system
- No agent-to-agent direct replies
- Comments are labels only: A or B
- Private opinion flip is a toggleable mechanism, off by default
- All random draws go through a single seeded numpy RNG instance passed at initialization
- Results saved as CSV files in results/ folder; figures saved as PNG

---

## Expected Qualitative Behavior

If the model is working correctly:

- **Phase 1** (posts 1–20, all treatment conditions): B-agent participation and confidence should decline; A-agent participation and confidence should rise. The gap should widen across posts.
- **Control condition**: Both A and B agents should maintain participation near their baseline participation_tendency across all 40 posts with no systematic divergence between groups.
- **Treatment 1 (neutral Phase 2)**: B-agent participation should stabilize or recover slowly after post 20, but likely remain below the control baseline.
- **Treatment 2 (favor-B Phase 2)**: B-agent participation should recover, but how quickly and how completely is an open question the simulation should answer. A-agent participation may begin to decline.
- **Treatment 3 (favor-A continued)**: B-agent participation should continue declining across posts 21–40.

---

## What Would Make the Results Untrustworthy

- Results appear for only one random seed and disappear under others
- The control condition shows systematic divergence between A and B agents
- Ablation of social_sensitivity (set to 0) still produces suppression effects
- Confidence values cluster at 0 or 1 for most agents (update rule may be too aggressive)
- Treatment 1 and Treatment 2 produce identical trajectories (the two update pathways may not be functioning distinctly)
