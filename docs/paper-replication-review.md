# Course-correction review: replicating Herbster et al. (arXiv 2604.08169)

A review of `notebooks/sycophancy_steering.ipynb` (and the matching claims in
`README.md` / `src/syco_steering/`) against the actual paper, **"Activation
Steering for Aligned Open-ended Generation without Sacrificing Coherence"**
(Herbster, Zborowski, Tosato, Gidel & Tosato — arXiv 2604.08169v2, July 2026).

Format, per your learning style: for each finding, **the right answer first**
(what the paper actually does, with drop-in code where useful), then **why it's
right** and what to take away from it. Findings are ranked by impact on
replication fidelity. Section/table/equation references are to the v2 PDF.

Your deliberate scale-downs (1.5B model, sycophancy as the trait, string-proxy
metrics, free-Colab budget) are all reasonable and are **not** treated as
mistakes — they're listed at the end so the boundary between "wrong" and
"intentionally smaller" stays sharp.

---

## TL;DR

| # | Finding | Type | Impact |
|---|---------|------|--------|
| 1 | The paper's probe is trained on **response-averaged** embeddings, not per-token ones. The notebook's core rationale ("pooling would make the gate never fire") is inverted. | Misreading | High |
| 2 | The paper's headline results use **all-token steering** (prompt included). The notebook implements only response-token steering, which the paper shows recovers roughly **half** the trait score. | Misreading | High |
| 3 | The steering layer must be chosen by a **downstream steering sweep**, not by probe accuracy. The paper considers this one of its key methodological points. | Known deviation, but under-weighted | High |
| 4 | α grids don't match the paper, and switching to pooled calibration (finding 1) **changes what α means** for StTP. | Consequence of 1 | Medium |
| 5 | Generation config: paper samples at T=0.6, top-p=0.9, up to 1024 tokens; notebook greedy-decodes 64. | Divergence | Medium |
| 6 | The paper varies the contrastive **system prompts across scenarios** (5 variants for dismissiveness; per-scenario for dishonesty); the notebook uses one global pair, maximizing the "probe learns the prompt, not the trait" confound. | Divergence | Medium |
| 7 | Two of the paper's **judge-free metrics** (cross-entropy vs. the aligned model, embedding distance) fit a free-Colab budget and would upgrade the string-proxy eval substantially. | Missed opportunity | Medium |
| 8 | Small factual errors: comparison model is **Qwen3.6-27B**, not Qwen3-32B; paper trains on 50 scenarios and evaluates on 112 (dishonesty) / 40 (dismissiveness). | Nits | Low |

Things the notebook gets **right** are listed at the end — there are several,
and some are subtle (the `m = −b/‖w‖` algebra, the hook indexing, group-aware
splits, the gate-firing diagnostic).

---

## Finding 1 — The probe is trained on response-averaged embeddings (the notebook has this backwards)

**What the paper does (§3.2, Eq. 1).** For each of the N training scenarios it
generates one on-policy response pair (a⁺ᵢ, a⁻ᵢ), then collects **one embedding
per response**: the *mean* hidden state over that response's tokens at layer ℓ:

> E⁺ₗ = { h̄ₗ(pᵢ, a⁺ᵢ) }ᵢ₌₁..N   and   E⁻ₗ = { h̄ₗ(pᵢ, a⁻ᵢ) }ᵢ₌₁..N
>
> "where h̄ₗ denotes the **mean hidden state over response tokens** at layer ℓ"

The logistic regression (L-BFGS, C = 1.0 — Table B.2) is fit on those N + N
pooled examples (50 + 50 in the paper). Everything downstream — v̂, Δμ, m, μ⁺,
σ⁺ — is computed from **response-averaged** projections (Alg. A.1 lines 3–7).
Only the *inference-time gate* is per-token: the runtime hook projects each
token's hidden state and compares it to m (Alg. A.1 "Runtime steering hook").

**The notebook/README claim the opposite** (notebook cells 0, 3, 11; README
§Method item 2): that the probe must be fit per-token because "a boundary fit
on pooled (mean) activations sits in a far tighter distribution, so the gate
would essentially never fire." The premise about the tighter distribution is
true; the conclusion doesn't follow, and the paper is direct evidence against
it — it calibrates pooled and gates per-token, and the gate demonstrably fires
(Fig. 1 bottom-right "All Tokens Projection" panels show the per-token
distributions straddling mₗ; Fig. A.2 annotates mₗ against those distributions
at four layers; §5.2 even attributes StTP's capability loss on Llama honesty to
the gate firing *too often* on neutral tokens).

**Why pooled calibration still gates per-token correctly.** Averaging T token
vectors shrinks the *within-class spread* (roughly by √T for the independent
component), but it does **not move the class means**: the mean of per-token
projections equals the (length-weighted) mean of per-response averages. The
logistic boundary m lands *between* the two class means. When you then project
individual tokens, their distributions are much wider — but they're centered on
those same two means, so a large fraction of misaligned tokens still falls
below m and a large fraction of aligned tokens stays above it. The gate fires;
it's just a noisier classifier per token than per response, which is exactly
what the paper's Fig. 1 histograms show (clean separation of averages, partial
overlap of tokens).

The quantity that *is* radically different between the two calibrations is
**σ⁺**: the per-token σ⁺ is several times the pooled σ⁺. That doesn't break the
gate (σ⁺ isn't in the gate), but it silently rescales StTP's target
sₗ = μ⁺ + α·σ⁺ — see finding 4.

There's also a real statistical cost to the per-token fit that the notebook
absorbs without flagging: response tokens are heavily dominated by generic
material ("the", punctuation, restating the question) that carries no stance,
so the token-level LR fits a boundary partly determined by noise tokens of both
classes. The pooled fit averages that noise out per response and recovers
cleaner class geometry from far fewer examples. This is the standard
probing-literature trade-off (compare mean-pooled probes in RepE, Zou et al.
2023, and CAA's response-averaged vectors, Rimsky et al. 2024).

**The right answer, as drop-in code.** You already have per-token activations
with group labels, so pooling is a five-line change — you can keep the
extraction cells untouched and add this next to `extract_direction`:

```python
def pool_by_response(X, groups, layer):
    """One response-averaged embedding per generation (paper Eq. 1)."""
    ids = np.unique(groups)
    return np.stack([X[groups == i, layer].astype(np.float32).mean(0) for i in ids])

def extract_direction_pooled(X_h, X_s, g_h, g_s, layer):
    E_pos = pool_by_response(X_h, g_h, layer)          # honest, (N, d)
    E_neg = pool_by_response(X_s, g_s, layer)          # sycophantic, (N, d)
    X = np.concatenate([E_pos, E_neg], 0)
    y = np.concatenate([np.ones(len(E_pos)), np.zeros(len(E_neg))]).astype(int)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)   # lbfgs is sklearn's default
    w = clf.coef_[0]; b = float(clf.intercept_[0]); n = float(np.linalg.norm(w))
    v_hat = (w / n).astype(np.float32)
    m = -b / n                       # identical algebra to Alg. A.1 (the Δμ rescaling cancels)
    proj_pos = E_pos @ v_hat
    proj_neg = E_neg @ v_hat
    delta_mu = float(proj_pos.mean() - proj_neg.mean())
    return {
        "v_hat": v_hat, "m": float(m),
        "mu_pos": float(proj_pos.mean()), "sig_pos": float(proj_pos.std()),
        "delta_mu": delta_mu, "best_layer": int(layer),
        "steering_vector": (v_hat * delta_mu).astype(np.float32),   # v = Δμ·v̂, ‖v‖ = Δμ
    }
```

The runtime hook needs **no change** for this: it already projects each token
and compares to `m`, which is exactly the paper's runtime loop.

Two knock-on effects to be aware of:

- **The token-level layer sweep stops working as-is for the pooled probe.**
  With ~60+60 pooled examples in 1536 dimensions, LR will hit ~100% held-out
  accuracy at nearly every layer, so accuracy can't rank layers. That's not a
  bug in pooling — it's the reason the paper doesn't select its layer by probe
  accuracy at all (finding 3). If you want a cheap probe-side diagnostic per
  layer, use the **Cohen's d of the pooled projections** (the paper annotates
  exactly this per layer in Fig. A.2), but treat it as a diagnostic, not the
  selection criterion.
- Your per-token projection histogram (cell 7) and gate-firing diagnostic
  (cell 12) become **more** informative, not less: they now test the actual
  paper mechanism (pooled boundary vs. token distribution) instead of testing
  the probe on its own training distribution.

**Keep the per-token probe as a documented *comparison*, if you like** — it's a
defensible variant and contrasting the two calibrations on the same activations
(where does m land? how do the gate-firing rates differ?) would be a genuinely
interesting notebook section. But the paper's method is the pooled one, and the
README/notebook prose asserting the paper is per-token should be corrected
either way.

---

## Finding 2 — The paper's headline results steer ALL tokens, prompt included

**What the paper does (§3.3 "Steering Position", §C.1, Table C.1).** Every
method supports two position modes: `all` (every position, *including the
prompt encoding during prefill*) and `response` (generated tokens only). The
main-text results (Fig. 3), the chosen operating points, and **all** downstream
experiments (multi-turn, MASK, Among Us, AuditBench, emergent misalignment) use
`all` mode. Response-only steering is the ablation, and it costs a lot:

| Llama-3.3-70B, dishonesty threat | trait (all) | trait (response) |
|---|---|---|
| SwFC (layer 26, α=4) | **75** | 38 |
| StTP (layer 26, α=36) | **77** | 42 |
| StMP (layer 26, α=4) | **62** | 42 |

(aligned baseline 93, malicious baseline 24 — Table C.1. The same pattern
replicates on Qwen: §D.2 reports peak honesty dropping from ~80 to ~50.)

The notebook implements only the equivalent of `response` mode — the hook masks
prefill edits to the final position. So even with everything else perfect, you
were running the configuration the paper shows recovers roughly **half** of the
recoverable trait. If your steered results looked underwhelming next to the
honest baseline, this is the single most likely cause.

**Why all-token steering wins.** The paper's stated hypothesis (§D.2): steering
the *prompt-encoding* activations attenuates the malicious system prompt's
influence **before generation begins**. In `response` mode the poisoned prompt
representation sits untouched in the KV cache, and every generated token
attends back into it — you're bailing water while the leak is still open. In
`all` mode the system prompt's own hidden states at layer ℓ get pushed across
the boundary too, so the cached keys/values that generation conditions on are
already partially detoxified. This connects to the Qi et al. (2025) motivation
in the introduction: alignment interventions concentrated at a few positions
are shallow; the fix is to intervene everywhere, continuously.

**The right answer, as drop-in code.** Add a `position` argument and make
`"all"` the default (matching the paper); keep `"response"` for the ablation:

```python
class Steer:
    def __init__(self, model, geom, method, alpha=None, layer_idx=None,
                 position="all"):                     # paper default; "response" = ablation
        ...
        self.position = position

    def _edit(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        h = hs.float()
        rho = h @ self.v
        ...compute add as before...
        if self.position == "response" and hs.shape[1] > 1:
            mask = torch.zeros_like(add)
            mask[:, -1] = add[:, -1]
            add = mask
        # position == "all": edit every prefill position and every decode step
        edited = (h + add.unsqueeze(-1) * self.v).to(hs.dtype)
        return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited
```

One batching caveat: with left padding, `"all"` mode also edits pad positions.
That's harmless — pad positions are masked out as attention keys, so their
edited states influence nothing — but worth a comment in the code so the next
reader doesn't puzzle over it.

Then make the position mode part of your comparison table (aligned baseline /
misaligned baseline / each method × {all, response}). Reproducing the paper's
*all ≫ response* ordering at 1.5B scale would itself be a nice replication
result — it's one of the two findings the paper says generalizes across
architectures (§D.3).

---

## Finding 3 — Layer selection is a downstream decision, not a probe decision

The notebook documents this deviation (Part B intro, README limitations), so
this is less "mistake" than "under-weighted": the paper treats it as one of its
**central claims**, not an implementation detail. From the Discussion (§6):

> "Wu et al. (2025) found that simple prompting baselines outperform
> representation-based steering... **We attribute much of this gap to the
> quality and specificity of the training data used to compute the steering
> direction, and selecting the right layer for intervention via the layer
> sweep.**"

**What the paper does (§4.3 "Operating Points", Fig. 3, Table C.1).** For every
layer (all 80 on Llama, all 64 on Qwen), every coefficient in the method's
grid, and both position modes, it generates steered outputs on the test
prompts and scores trait + coherence with the judge. The operating point
maximizes trait **subject to coherence ≥ 90% of the aligned baseline's
coherence**. Probe accuracy never enters the selection.

Two facts that make probe-selected layers a poor proxy:

- On Llama the steering-optimal layers are **26–29 of 80 (~29–40% depth)**; on
  Qwen they sit around **50% depth** (§D.1: "layer selections cannot be
  transferred across architectures and a per-model sweep is required"). Probe
  separability, by contrast, typically keeps improving into the mid-late
  layers (the paper's own Fig. A.2 shows Cohen's d peaking around layer 32 and
  *plateauing* — while steering quality at layer 48+ falls off). Where you can
  *read* a trait best is not where *writing* it steers generation best: edits
  at late layers have few remaining blocks through which to influence the
  computation, and edits at probe-optimal layers can be off the model's
  natural data manifold.
- SwFC at its optimum looks competitive, but the paper stresses the optimum is
  **brittle** — "performance degrades sharply with small changes in layer or
  coefficient" (§6). If you only evaluate at one probe-chosen layer you can't
  see any of this structure.

**Budget version for Colab.** You don't need 80 × 7 × 2 judge calls; you need a
small grid and cheap scores:

1. Pick 4–6 candidate layers spanning ~25%–60% relative depth (for a 28-layer
   Qwen2.5-1.5B: hidden-state indices ≈ 7, 10, 13, 16, 19).
2. For each candidate layer × each α in the method's grid, generate on your
   K_SWEEP prompts and score with your existing proxies **plus** the two
   paper-native cheap metrics from finding 7 (cross-entropy is the paper's own
   coherence-without-a-judge signal).
3. Select by trait proxy subject to the coherence constraint (the paper's
   ≥ 90%-of-aligned-baseline rule maps naturally onto your repetition budget +
   cross-entropy staying near the aligned baseline's own value).
4. Plot trait-vs-layer per method, like Fig. 3. Even a coarse version of that
   plot is the paper's signature figure and will teach you more than the
   single-layer numbers.

Your existing per-token sweep can stay as a cheap *shortlist* generator for
step 1 — just don't let it make the final call.

---

## Finding 4 — α grids and α semantics (units changed under your feet)

Paper grids (Table B.2) and chosen operating points (Table C.1):

| Method | Paper grid | Paper optimum (honesty / compassion, all-token) | Notebook grid |
|---|---|---|---|
| SwFC | α ∈ {1, 2, 3, 4, 5}, h′ = h + α·**v** with ‖v‖ = Δμ | 4 / 2 | {0.5, 1, 2} in Δμ units — equivalent parameterization ✔, range too small |
| StTP | α ∈ {0, 6, 12, 18, 24, 30, 36}, target s = μ⁺ + α·**σ⁺** | 36 / 24 | {0.5, 1, 2} |
| StMP | α ∈ {1, 1.5, 2, 2.5, 3, 3.5, 4} (unit-free mirror factor) | 4 / 3 | {1, 1.5, 2} |

- **SwFC**: your `h + α·v̂` with α defaulting to Δμ is *exactly* the paper's
  α = 1 (their v is already Δμ·v̂ — §3.2: "α = 1 shifts activations by one
  natural unit of class separation"). Correct reading on your part; just widen
  the grid to {1..5}·Δμ, since the paper's optima are 4Δμ and 2Δμ.
- **StTP** is where finding 1 bites: α multiplies **σ⁺**. The paper's σ⁺ is the
  std of *response-averaged* projections — small — so pushing meaningfully past
  μ⁺ needs α up to 36. Your per-token σ⁺ is several times larger, so your
  α = 1–2 may already correspond to a mid-sized paper α — or not; you can't
  tell without converting units. Once you switch to pooled calibration
  (finding 1), adopt the paper's grid literally. Note α = 0 is in their grid:
  "clamp every below-boundary token exactly to μ⁺" is a meaningful setting.
- **StMP**: α is a pure interpolation factor (1 = exact reflection across m),
  so it's unit-safe under any calibration — but the paper's optima (4 for
  honesty, 3 for compassion) are *above* your grid's ceiling of 2. Extend to 4.

Takeaway worth internalizing: **a steering coefficient only means something
relative to the statistics it multiplies.** Whenever you change how μ, σ, or Δμ
are estimated, every α in every grid changes meaning simultaneously. Recording
the *absolute* projection displacement (the paper's "target distance" metric,
§C.3) alongside α is the way to keep results comparable across calibrations.

---

## Finding 5 — Generation configuration

Paper (Table B.2): **temperature 0.6, top-p 0.9, seed 42, max 1024 new tokens**
for both training generations and evaluation. Notebook: greedy, 64 tokens.

Why it matters, in order:

1. **Truncation at 64 tokens** hits the probe's training data and the eval
   distribution alike. Sycophancy is front-loaded ("You're right!"), so the
   probe survives, but the eval side is where the paper's whole contribution
   lives — coherence, repetition, and degeneration are properties of *long*
   generations (their token-count metric runs to ~1000 tokens, and SwFC's
   repetition pathology only develops at length). At 64 tokens you're mostly
   blind to the failure mode that distinguishes StTP/StMP from SwFC. Even 256
   tokens on the eval side would sharpen this a lot on a T4.
2. **Greedy vs. sampling**: greedy is a reasonable budget choice for the
   *training* generations (you want the modal on-policy response), but for
   evaluation it collapses the output distribution to a single point —
   repetition proxies behave differently under greedy decoding (it loops more)
   and you lose any notion of variance across the eval set. If you keep greedy,
   at least note that your repetition numbers aren't comparable to sampled
   ones. Matching T=0.6/top-p=0.9 with a fixed seed is cheap.
3. Empty-generation edge case: with sampling, `MAX_NEW_TRAIN=64` truncations
   and occasional empty responses become more common; your extraction loop
   already skips empties — good.

---

## Finding 6 — Diversify the contrastive system prompts

Paper §4.1: the dishonesty training set is "50 scenarios, **each with a
contrastive system prompt**, a user prompt, and the target model's on-policy
honest and dishonest responses" — i.e., scenario-specific prompt pairs. The
dismissiveness set uses "**5 contrastive system prompt variants**" across its
50 prompts. The test side is 112 scenarios across 8 categories (dishonesty)
and 40 held-out prompts (dismissiveness).

The notebook uses **one** global sycophantic/honest pair for all 60 training
prompts. Your own README names the resulting risk precisely ("the probe may
partly encode *which system prompt is in context* rather than the trait") —
but with a single fixed pair, the system-prompt-identity feature and the trait
feature are *perfectly* confounded in your training data. The paper's design
doesn't eliminate this confound (you note that correctly), but varying the
surface form across 5+ paraphrase pairs forces the probe to find the component
that's *shared* across differently-worded inducements — much closer to "the
trait" than "the string".

Cheap fix: write ~5 paraphrase pairs of your sycophantic/honest prompts
(vary length, vocabulary, framing — e.g. one that never uses the word
"agree"), assign them round-robin across training prompts, and keep one pair
held out entirely for the eval-side threat prompt. If the probe direction
barely moves (CAA cosine between the single-pair and multi-pair directions),
that's evidence the direction was already trait-like; if it moves a lot, you
just caught the confound. Either outcome is a result.

---

## Finding 7 — Two of the paper's judge-free metrics fit your budget

You treat "LLM judge + ELO + capability suite" as monolithically out of budget
and fall back to string proxies. But the paper itself validates its judge with
two **cheap, judge-free** signals (§C.3, §C.5), both of which run fine on a T4
and would substantially de-risk your string-matching eval:

**(a) Cross-entropy under the unsteered model conditioned on the aligned
prompt** (§C.3, their coherence-without-a-judge metric): for a steered
response y generated under the malicious prompt, compute the unsteered model's
mean −log P(yₜ | y<ₜ, s⁺, q) over the first 50 tokens. If steering restored
honest-and-fluent behavior, the aligned model should find the text *likely*;
degenerate or off-policy text scores high. Compare against the same quantity
computed on the aligned baseline's own responses (their "matched baseline").

```python
import torch.nn.functional as F

@torch.no_grad()
def ce_vs_aligned(user, response, n_tokens=50):
    """Paper §C.3: mean cross-entropy of `response` under the UNSTEERED model
    conditioned on the aligned (honest) system prompt, first n_tokens."""
    msgs = conv(SYSTEM_HONEST, user)
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    f_ids = tok.apply_chat_template(msgs + [{"role": "assistant", "content": response}],
                                    add_generation_prompt=False, return_tensors="pt")
    p_ids, f_ids = p_ids.to(model.device), f_ids.to(model.device)
    plen = p_ids.shape[1]
    logits = model(f_ids).logits[0, plen - 1 : -1].float()   # predict response tokens
    targets = f_ids[0, plen:]
    k = min(n_tokens, targets.shape[0])
    return float(F.cross_entropy(logits[:k], targets[:k]))
```

This directly replaces the weakest part of the current eval: `rep3` catches
loops but not incoherence, and the string proxies catch neither.

**(b) Embedding similarity to the aligned baseline** (§C.5): embed each steered
response and its aligned-baseline counterpart with a small sentence encoder
(`sentence-transformers/all-MiniLM-L6-v2` is ~80 MB) and report cosine
similarity. The paper's logic: if steering genuinely recovers aligned
*behavior*, steered outputs should be measurably closer to the aligned
baseline's outputs than the malicious baseline's are — and the layer where
similarity peaks should agree with your other metrics (they use this
convergence as validation, §6). This also gives your budget layer sweep
(finding 3) a second, independent axis.

Also easy and worth stealing: the paper reports metrics on the **first 50
tokens** for all conditions "to ensure a fair comparison across conditions that
produce different response lengths" (§C.3). Your proxies currently run over
whatever length each condition happened to generate — truncate consistently.

If you later want one step beyond proxies without hosting a judge: a few
hundred pairwise judgments ("which response is more sycophantic?") through any
small API model is dollars, and pairwise choices are much more robust than
absolute 0–100 scoring — that's the entire reason the paper runs its ELO
tournament as a check on the judge (§4.3).

---

## Finding 8 — Small factual corrections

- The cross-architecture model is **Qwen3.6-27B (64 layers)**, not
  "Qwen3-32B" (README, notebook cell 34).
- The paper's judge is gpt-oss-120b at **temperature 1.0, high reasoning
  effort** (Table B.1); its generation seed is 42, LR is L-BFGS with C = 1.0
  (Table B.2) — your `LogisticRegression(C=1.0)` (lbfgs default) matches.
- Training scenarios: **50 per trait**; eval: **112** (dishonesty, 8
  categories) / **40** (dismissiveness). Your 60/25 split is the right shape,
  just noting the reference numbers.
- Paper baselines worth keeping in mind as a sanity anchor (Table C.1,
  honesty): aligned 93, malicious 24, best steered (StTP all, α=36) 77 —
  i.e., even at 70B with the full method, steering recovers *most but not
  all* of the gap. Calibrate your expectations for 1.5B accordingly.

---

## What the notebook already gets right

Worth naming, because several of these are easy to get wrong:

- **On-policy contrastive generations** (the model's own responses under each
  prompt) — exactly the paper's §3.2 design, and the thing the paper credits
  for beating prior negative results ("quality and specificity of the training
  data").
- **The boundary algebra.** Your `m = −b/‖w‖` is exactly what Alg. A.1's
  rescaling reduces to (b′ = b·‖v‖/‖w‖, m = −b′/‖v‖ ⇒ −b/‖w‖). The
  paper's Δμ-rescaling detour cancels out; you implemented the fixed point of
  it directly.
- **All three intervention formulas** (SwFC / StTP / StMP) match Eq. 3–5,
  including StMP's 2α(m−ρ) reflection and StTP's (s−ρ) projection-set.
- **Hook indexing**: `hidden_states[l]` = output of `model.model.layers[l-1]`,
  and the hook edits precisely the tensor the probe was trained on. The
  README's index-convention note is correct and appreciated.
- **Group-aware splits** (whole responses on one side) — this is the right
  defense against token-correlation leakage, and it's *not* in the paper
  (pooled probes don't need it); keep it for any per-token analysis.
- **Disjoint α-sweep / final-eval prompts**, and evaluating under the active
  threat (malicious prompt on) against both baselines — matches §3.1's
  formulation of what "restores alignment" means.
- **The gate-firing diagnostic** (cell 12) has no direct counterpart in the
  paper and is genuinely good practice — it's the instrument that would have
  caught finding 1 empirically, and it directly measures the mechanism behind
  the paper's own StTP-on-Llama capability regression (§5.2: neutral tokens
  projecting below m). Keep it after recalibrating.
- **CAA-cosine consistency check** mirrors the paper's §A.3 (mean cosine
  0.94–0.99 where classes separate well; 0.80 for Qwen honesty). Your 0.80
  threshold is well chosen — and §A.3 gives you the interpretation for a *low*
  value: when classes overlap, LR rotates away from the centroid difference to
  maximize discrimination, which is an argument *for* the LR direction, not
  against it.

---

## Suggested order of changes

1. **Recalibrate the probe on response-averaged embeddings** (finding 1) —
   `pool_by_response` + `extract_direction_pooled`; keep the per-token version
   as an explicit comparison if you want, and fix the README/notebook prose
   that attributes per-token training to the paper.
2. **Add `position="all"` to the hook and make it the default** (finding 2) —
   this is the highest expected impact on your steered numbers.
3. **Adopt the paper's α grids** ({1..5}·Δμ / {0,6,…,36} / {1,…,4}) now that
   σ⁺ and Δμ are in the paper's units (finding 4).
4. **Add cross-entropy + embedding-distance metrics, truncate all metrics to
   the first 50 tokens** (finding 7).
5. **Run a small downstream layer × α grid** (4–6 layers at 25–60% depth)
   selecting by trait-proxy subject to the coherence budget; plot
   trait-vs-layer per method (finding 3).
6. Lower priority: sampled decoding at T=0.6/top-p=0.9 with longer eval
   generations (finding 5); 5 paraphrased system-prompt pairs (finding 6);
   factual nits (finding 8).

Items 1–4 are each small, local edits to the existing notebook; item 5 is the
only one that costs real compute (a few extra sweep-sized generation passes —
budget roughly an additional 20–40 T4-minutes at K_SWEEP=8 with 4 layers × 3
coefficients per method).
