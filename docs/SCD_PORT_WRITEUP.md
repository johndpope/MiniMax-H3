# Separable Causal Diffusion on MiniMax H3: what is established, and what is still a bet

**Status as of 2026-08-11.** Phases 0–2.5 are done and measured. Phase 3 finished its first
training run and **passed its kill criterion** — the split generates structured video latents
that correlate +0.61 with ground truth under autoregressive rollout, against +0.12 for the
untrained split. It has been validated **only in latent space, only on training clips, only at
512², and only over 1.2 seconds.** No pixels have been decoded. Read section 8 before quoting
anything here.

---

## 1. The idea in one paragraph

A video diffusion transformer runs every layer over every frame, N times, once per denoising
step. Most of that is waste: the early layers are mostly working out *what is in the scene* and
*how it moves*, and that answer barely changes between step 3 and step 4 of the same frame.
Separable Causal Diffusion (SCD, arXiv:2602.10095) splits the stack in two. An **encoder**
prefix runs **once per frame**, on clean latents, with a frame-causal mask and a KV cache — it
never sees the noise, so there is nothing for it to re-do. A **decoder** of a handful of layers
runs **N times per frame**, on the noisy latents, reading the encoder's output as a prefix.
The bet is that the expensive-and-repeated part and the cheap-and-repeated part can be pulled
apart with the quality mostly intact.

If it works, two things follow. Multi-step generation gets cheaper roughly in proportion to N.
And, more importantly, peak memory stops growing with clip length, because the decoder's
attention is over one frame plus a bounded cache rather than over the whole clip.

## 2. The honest summary, up front

| Claim | Status |
|---|---|
| H3 is a better host for SCD than the model the technique was demoed on | **Measured.** §3 |
| The split point is not arbitrary — two independent measurements agree on it | **Measured**, with one caveat that is stated in §4 |
| The chunked, cached, windowed encoder is *bit-exact* against one masked pass | **Proved by test**, `torch.equal` |
| The windowed encoder is an approximation, and what it costs is known | **Measured**, and it costs video almost nothing and audio a lot |
| The split graph is 2.5–3.9× faster than stock H3 on real weights | **Measured**, with the range being a design decision not yet made |
| Stock H3 cannot generate past ~2 s at 768p on a 24 GB card; the split can do 5 s+ | **Measured.** This is the stronger result |
| A 33B model can be adapted for this on **one** 24 GB consumer card | **Running.** 15.4 GB peak, 7.7–13.7 s/step |
| The trained split produces structured latents, not noise, and beats the untrained split 4.9× | **Measured.** §8 |
| The autoregressive rollout does not compound error over 7 frames | **Measured**, and it was the expected failure. §8 |
| **The split produces watchable video** | **Unknown. No pixels have been decoded.** |
| **It generalizes beyond the 12 clips it was fit to** | **Unknown. Not tested.** |

Everything above the last line is engineering, and engineering can be checked. The last line is
the research question, and it is open.

---

## 3. Why H3, and why the technique needed a bigger model than it was demoed on

SCD's win comes from moving work out of the per-step loop. That only pays if the per-step work
is dominated by something the encoder can absorb — in practice, by attention, which is quadratic
in sequence length. Below some sequence length, attention is a rounding error and the split buys
nothing.

We measured that crossover directly (`scripts/scd/tier0_bench.py`, weights-free, RTX PRO 4000
Blackwell, bf16, N=16 denoise steps):

| configuration | sequence | attention share of FLOPs | speedup |
|---|---|---|---|
| 512², T=4 | small | 3.7% | **1.41×** |
| 768p, 5 s | ~21k | 53.8% | **2.67×** |
| 768p, 10 s | ~41k | 69.6% | **3.90×** |
| 768p, 15 s | ~62k | 77.3% | **5.03×** |

The crossover — where attention becomes half the cost — is at **26.9k tokens**. LTX-2, the model
the reference implementation targets, packs 336 tokens per frame and never reaches it at any
practical length. H3 at 768p packs **1008 rows per latent frame** and sits 1.2–3.4× past it.
So the technique is not merely portable to H3; H3 is the regime it was designed for and LTX-2
was not. The win also **grows with N**, which matters because any later step-count increase to
recover quality is partly self-funding (3.50× at N=8, 4.12× at N=30 for 768p/10s).

These are FLOP-model microbenchmarks with no weights loaded. Measured wall-clock tracked the
model within 4%, but they are not end-to-end generation numbers. Section 6 has those.

## 4. Where to cut the stack

H3 has 50 transformer blocks. The question is which are encoder and which are decoder, and the
answer had to come from measurement because H3 was never trained to reason causally over time —
it is a bidirectional model, and the paper's separability result was measured on a model that
had already been converted to autoregressive. That gap is the single largest assumption in the
whole port and it is why the split point was measured twice, from two directions.

**Measurement A — where do features stop being noise-invariant?** If a block's output is nearly
the same at σ=0.25 and σ=0.9, that block is not doing denoising work and can be amortized.
Cosine similarity across noise levels on text rows is flat at 0.973–0.975 from block 13 all the
way to 27, then falls: 0.960 (28), 0.910 (29), and collapses to **0.777 at block 30**. That
knee bounds the encoder at **30 blocks**.

**Measurement B — which blocks hurt most to remove?** This is the paper's own criterion, run on
H3: delete one block, measure the loss. Block **0 costs +96%**, block **49 +82%**, block
**1 +55%**, block **48 +4.5%** — and *all 46 others* land within ±0.03 and sum to −0.06. The
model has two ends that matter and a long cheap middle. That forces a decoder built from **both
ends**: `{0, 1} ∪ {45–49}`, seven blocks, matching the paper's own "first 5 and last 5" shape.

**Measurement C, and it arrived independently.** When we later measured what the frame-causal
mask *costs* — scoring every block's output under the causal mask against the unmasked pass on
real weights — the video curve is flat at 0.998–0.999 from block 1 through block 25, reads 0.952
at block 29, and then falls off a cliff: 0.794 (30), 0.471 (35), 0.412 (45). **The mask is nearly
free over exactly the span the encoder occupies and expensive immediately past it**, and the
knee lands on the same block as measurement A, from a completely unrelated experiment.

So: encoder = blocks 0–29, decoder = blocks {0, 1, 45–49}, 37 layer instances from a 50-block
model. Blocks 0 and 1 are instantiated twice. There is therefore **no identity path** — the
composed split cannot reproduce the base forward by construction, which is a property of the
technique, not a bug in the port.

**The caveat, stated plainly.** Measurement A's *video*-row knee is **not** independent
corroboration of its text-row knee: it tracks its own threshold parameter (30, 30, 30, 29, 26,
25 as the threshold sweeps), so it can be made to say what you want. Only the text knee survives
a sweep of its own threshold. Measurement C is genuine independent support; the video knee is
not, and was nearly written up as though it were.

**One measurement had to be thrown away and redone.** The first leave-one-out used a flat grid
of noise levels and produced a baseline *worse than predicting zero* (cosine −0.44 at σ=0.1).
H3 does not train on a flat grid: it samples σ as `12u/(1+11u)`, median ~0.92, with only ~3% of
steps below σ=0.3. Re-running on the model's own density moved video-row σ-invariance from 0.061
to **0.779** — an order of magnitude. The knee index did not move, so the conclusion survived,
but the number that would have been quoted was meaningless.

## 5. Making the encoder exact, then making it bounded

An encoder that runs once per frame with a KV cache is only useful if it computes the same thing
as one masked pass over the whole clip. Three transformations were needed, and the first two are
**exact** — the same arithmetic, reassociated — while the third is not, which is stated as a
distinction rather than blurred.

**Chunking with a KV cache is exact.** `test_chunked_matches_full` asserts `torch.equal`, not
a tolerance.

**The 53 GB problem was a loop order, not a property of the encoder.** Running chunks-outer /
blocks-inner — the obvious order, and the one a streaming AR driver is forced into — keeps all
30 block caches live at once: 840 KiB per row, **53 GB** at 768p/15 s. Transposing the loops
(`layer_major=True`) means only *one* block's cache is ever live: 28 KiB per row, **1.8 GB**.
Same arithmetic, because block *i*'s input on chunk *c* is block *i−1*'s output on chunk *c* and
nothing else. This also cuts weight transfers under block-swapping from 2760 to 30, which is
what would otherwise have turned the encoder into a PCIe benchmark and reported SCD as slow for
reasons unrelated to SCD.

**Audio needed its own clock.** H3 generates native 32 kHz stereo alongside video. Under the
first mask design, audio rows were treated as context — exempt from time order — which meant the
encoder's audio rows could see text and audio and *no video at all*. Measured, audio's centered
correlation with the bidirectional reference at the encoder output was **0.047, and −0.074 at
higher σ**: uncorrelated, and slightly anticorrelated. The raw (uncentered) cosine read 0.980
and 0.996, which looks like health; common mode on those rows is 0.996, so the raw number was
measuring the shared vector and essentially nothing else. **That was one edit away from being
written into the design as a finding.** Putting audio on its own 40 Hz clock — audio at time *t*
sees video ≤ *t* and vice versa — moves audio to **0.445 / 0.339** and costs video about 0.03 of
centered cosine. It is a trade, and it is described as one.

**The window is the one approximation.** Evicting cache rows beyond the last W frames is what
makes memory flat in clip length, and no test can pin what it costs, so it was measured against
the same path at `window=None`. At real 768p geometry:

| | video | audio |
|---|---|---|
| window=8 | 0.997 | 0.859 |
| window=12 | 0.999 | 0.906 |
| window=20 | 1.0000 | 0.992 |

And the per-frame breakdown separates two things a pooled average cannot: **video's error is
bounded** (it asymptotes just under 0.99 past frame W and stops) while **audio's accumulates**
(it roughly halves across the post-window span at every width tried). Audio is 1.3% of the rows,
so exempting it from eviction entirely costs ~1% more cache and helps — 0.779 → 0.890 at
window=12 — but does not flatten the curve, because the problem is that the *video* history
those audio rows attend to is still being evicted. That is a named, quantified, unfixed
limitation, not a solved one.

## 6. Does it actually go faster — on real weights

`docs/phase25_tier1.json`. Real FL2VA weights at NF4, 768p, encoder 0–29, decoder {0,1,45–49},
`window=12`, `layer_major=True`. Each stage timed with only its own blocks resident.

| latent frames | duration | stock s/step | stock peak | encoder (once) | decoder ms/frame | dec peak |
|---|---|---|---|---|---|---|
| 12 | 1.9 s | **11.219** | 19.4 GB | 10.215 s | 212.9 | 3.7 GB |
| 22 | 3.5 s | **OOM** | — | 21.649 s | 215.8 | 3.8 GB |
| 32 | 5.2 s | **OOM** | — | 32.985 s | 214.3 | 3.9 GB |

**Speedup at the one length where both sides run: 2.93× (N=8), 3.51× (N=16), 3.87× (N=30).**
The theoretical ceiling as N→∞ is 4.39×.

**The result that is not a ratio is the better one.** Stock H3 OOMs at 3.5 s on this card — at
768p it cannot denoise a clip longer than about **two seconds** on 24 GB, because its attention
spans the whole packed sequence and that grows 1008 rows per frame. The split ran 5.2 s at
13.1 GB with headroom. Past two seconds the comparison is not fast-versus-slow, it is
running-versus-not-running.

Three things scale as designed, and one does not:

- **KV cache is flat:** 12758 → 12778 → 12796 rows while the sequence grows 12758 → 33184. An
  unbounded cache would grow 1008 rows per frame. The 20-row wobble is audio landing on its own
  grid, not leakage.
- **Decoder is flat per frame:** 212.9 / 215.8 / 214.3 ms, 0.7% spread across a 2.6× length
  span. This is the load-bearing one for long video.
- **Encoder is linear, not quadratic:** +11.4 and +11.3 s per 10 frames. The window is what makes
  it a line.
- **Encoder peak VRAM is *not* flat:** 12.1 → 12.6 → 13.1 GB. This is not the cache (pinned at
  0.37 GB) — it is the packed hidden state itself, `[S, 5376]` in bf16, which must be resident
  because the layer-major encoder rewrites the whole sequence per block. So "flat VRAM" is true
  of the cache and only approximately true of the encoder.

**Two honesty notes on this table.** First, at 12 latent frames the clip is *shorter than the
12-frame window*, so the encoder in the row that produced those speedups ran **unbounded** —
every row where the window actually evicts is a row where stock OOM'd, so no single row both
prices a windowed encoder and has a baseline. Re-running at `window=8`, where eviction does fire,
gives 2.96 / 3.54 / 3.89× — unchanged, and the windowed encoder is marginally *faster*. Second,
the decoder timed here packs 2R rows and the design specifies 2R + text; the text version is
1.26× slower per frame, taking N=30 from 3.90× to **3.18×**. So the honest headline is a **range,
2.5–3.9×**, whose width is a design decision that cannot be made until a trained model exists to
A/B.

## 7. Fitting a 33B model on one 24 GB card

The training loop runs at **15.4 GB peak, 7.7–13.7 s/step**. Getting there cost two changes and
one wrong hypothesis, and the wrong hypothesis is the part worth recording.

The loop OOM'd at 19.2 GB. Activations were the obvious suspect, so gradient checkpointing went
in — correctly, at one region per block spanning all chunks, because the KV cache is a forward
side effect and a finer region would re-enter a half-filled cache. It moved the peak to 18.86 GB.
Measuring instead of guessing gave the actual budget: base loaded 16.06 GB, after the split
12.02 GB, after rank-32 LoRA 12.52 GB, on ~22.2 GB usable. **Weights, not activations.**
Checkpointing stayed because it is correct and cheap, but it was not the fix.

The two changes that were:

- **Rank 16, not 32.** Rank is charged four times over — factors, gradients, and two AdamW
  moments, all fp32. 155 adapters at rank 16 are 66.3 M parameters against 12 clips; capacity
  was never the binding constraint.
- **A two-stage backward.** All seven frames read the same encoder output, which leaves a choice
  between holding seven decoder graphs to the end of the step or freeing the encoder graph on the
  first frame. Detaching the encoder output into a leaf gets neither: each frame's graph dies as
  it is scored, its gradient accumulates on the leaf, and the encoder is replayed once with the
  sum. Gradients are **bit-identical** to the one-shot version — the same sum, associated
  differently.

That bit-identity is exactly why the memory property needs its own test. `backward(retain_graph=
True)` gives numerically identical gradients and OOMs on the real card, so no gradient comparison
can distinguish them. The test counts instead: every tensor autograd saves is wrapped, a weakref
fires when its graph dies, and peak-live/total reads **0.46 cut against 0.99 retained**.

### A note on how this was built

Every non-obvious claim in this document is pinned by a test, and every test was checked by
deliberately breaking the code it guards. 56 unit cases across five suites, and each case was
justified by a mutant it and nothing else catches. Two mutants **survived** their first suite,
and diagnosing why corrected beliefs rather than code:

- A square test fixture cannot detect a transposed `unpatchify`. The real clips are 32×32, so
  training would not have caught it either. The fixture is now 8×12.
- Seeding the autoregressive rollout's context buffer from ground truth *looks* like the classic
  evaluation leak and is actually harmless, because the encoder is frame-causal: by the time
  frame *f* is decoded, every frame below it has been overwritten by generated content, and
  ground truth above it cannot reach the decoder. The dangerous version is that change *together
  with* a missing write-back. The test now pins the combination.

The single most valuable test is `test_a_perfect_velocity_recovers_the_clip`: feed the sampler
the true velocity and it must return the true latent to 1e-5. A sign-flipped flow-matching
convention does not crash — it produces a latent with a sane standard deviation and near-zero
correlation, which is **indistinguishable from "the split did not learn"** and would have been
reported as a kill.

## 8. The quality result

### How it was going to be read, written down before the numbers existed

This was committed to in advance and is quoted here unchanged, because "not pure noise" is
exactly the kind of criterion a run talks you into afterwards.

**Baseline is the unadapted split** — same module tree, same code path, adapters installed and
left at zero. Strictly harder than the phase's "decoder-only random init", and the honest bar: if
the trained adapters cannot beat what the surgery gives away free, the adaptation has not paid
for itself.

**Headline metric is correlation, not MSE**, on frames 1 onward. A decoder that collapses to the
mean posts a *lower* MSE than the truth's own variance while carrying no frame-specific
information; correlation is ~0 for anything uncorrelated with the target, including a confident
constant. Frame 0 is excluded because it has a zeroed context half by construction.

**Two modes, and the gap between them is the finding.** *Oracle* denoises with the encoder
reading real clean latents — the regime training ran in, a ceiling, not a result. *AR* is the real
rollout, where the encoder only sees frames the decoder itself produced. Oracle also flat → the
conditional was never learned, **a kill**. Oracle fine, AR collapses → exposure bias, the expected
consequence of training without scheduled sampling, and evidence **for** the curriculum rather
than against the split.

### What came back

2000 steps, 6.0 hours, one card. Training eval fell 2.3864 → 0.5408. Then 20 denoising steps per
frame, four clips, four ways:

| mode | adapter | **corr[1:]** | MSE | noise floor | pred std |
|---|---|---|---|---|---|
| oracle | **trained** | **+0.9039** | 0.2923 | 2.19 | 1.12 |
| oracle | unadapted | +0.1228 | 1.2283 | 2.19 | 0.405 |
| ar | **trained** | **+0.6064** | 0.8355 | 2.35 | 1.16 |
| ar | unadapted | +0.1229 | 1.2292 | 2.35 | 0.403 |

**Not a kill.** The AR rollout is not noise and beats the unadapted split 4.9× on correlation.
The oracle result says the one-step conditional was learned; the bet that a rank-16 adapter
reaches a useful place on a 33B base holds at this scale.

**The baseline is the collapse trap, live.** Its MSE of 1.23 against a 2.19 noise floor reads like
partial success — and its prediction std is **0.403 against the truth's ~1.0**. It is shrinking
toward the mean, which lowers MSE while carrying nothing. Correlation says +0.12. Had the metric
been MSE, as the phase's own "not pure noise" wording invites, **the untrained split would have
scored as a partial pass.** The pre-registration earned its keep on the first run.

**The surprise is that AR does not compound.** Per-frame correlation, averaged over four clips:

| | f0 | f1 | f2 | f3 | f4 | f5 | f6 |
|---|---|---|---|---|---|---|---|
| oracle, trained | +0.444 | +0.882 | +0.893 | +0.907 | +0.911 | +0.915 | **+0.916** |
| **ar, trained** | +0.444 | +0.543 | +0.576 | +0.611 | +0.622 | +0.638 | **+0.648** |
| either, unadapted | +0.114 | ~+0.12 | ~+0.11 | ~+0.13 | ~+0.13 | ~+0.13 | ~+0.11 |

Both trained curves **rise monotonically**, and the oracle→AR gap *narrows* across the clip
(0.339 at f1 → 0.268 at f6). The AR penalty is a **level shift, not a slope**: the decoder pays a
one-time cost for conditioning on its own output and then stops degrading. That is the opposite of
the failure that was budgeted for.

It does not license skipping the scheduled-sampling curriculum. Seven latent frames is 1.2
seconds, and a slope too shallow to resolve over six frames is exactly what would still ruin a
15-second rollout. What it changes is the question: measure *where* the slope appears, not whether
it exists.

Two consistency checks fell out for free. Frame 0 reads **+0.444 in both modes to three decimals**
— the zeroed context half behaves identically at inference and at training. And the unadapted
split scores the same in AR as in oracle (0.1229 vs 0.1228), i.e. it ignores its context
completely, which is what makes the trained model's oracle→AR gap readable as conditioning rather
than as noise.

### What this does not establish, and none of it is a technicality

- **These are training clips.** 12 clips × 2000 steps at batch 1 is ~166 epochs. This measures
  that the split **can fit**, which is the Phase 3 question. Memorization is a live explanation
  for the oracle's 0.90 and nothing here rules it out.
- **No pixels exist.** Every number is latent-space correlation. Nothing has been through the
  video VAE; no human has looked at a frame. **+0.61 has no established mapping to perceptual
  quality**, and claiming one would be inventing a result.
- **1.2 seconds.** The no-compounding finding is measured over six frames.

### The named gaps, so they are not discovered later as surprises

- **512², not 768p.** 256 rows per latent frame against 768p's 1008. Nothing trained here
  validates the shipping geometry; it validates the mechanism at a quarter of the rows.
- **12 clips**, against the 20–50 the phase asks for.
- **No audio in the loss.** The clips carry video latents only. The audio window drift measured
  in §5 is something this run can neither improve nor refute.
- **No scheduled sampling.** Training is teacher-forced end to end. The model has never seen its
  own output as input.
- **LoRA, not full fine-tuning.** The paper adapts with a full fine-tune over 55K steps. Rank-16
  adapters over 2K steps on a much larger base is our extrapolation and should be labelled as
  one.

## 9. What this adds up to

The engineering case is made and measured: the split is exact where it claims to be exact,
approximate where it claims to be approximate with the cost quantified, 2.5–3.9× faster on real
weights, and — the stronger claim — it generates clip lengths on a 24 GB card that stock H3
cannot generate at all. The split point was not guessed; it was located by two independent
measurements that agree, one of which is the paper's own criterion re-run on this model.

The research bet — that a **bidirectional** model, adapted with a **light** adapter, retains
enough through a split that has no identity path — is no longer untested. It survived its first
contact with a sampler, and it survived in a specific and checkable way: 4.9× over a baseline
chosen to be hard, an oracle at +0.90 that says the conditional exists, and an AR rollout whose
error is a level shift rather than a slope. The last of those was the failure this run was
expected to produce, and it did not.

What has been shown is that **the mechanism works**. What has not been shown is that **the output
is good**. Those are different claims and the gap between them is one VAE decode wide: 12 clips
memorized at 512² over 1.2 seconds in latent space is not video, and until someone watches a
frame, +0.61 is a number whose perceptual meaning is unknown.

So the honest position is neither "it works" nor "it might not work". It is: the thing that could
have killed this cheaply did not, the next thing that could kill it costs one afternoon, and it
should be run before anything else is built on top.

---

*Reproduction: `scripts/scd/` (benchmarks, model surgery, train loop, sampler, five test suites).
Full method notes and the record of what was measured wrong first:
`docs/MINIMAX_H3_SCD_PORT_DESIGN.md`. Results as JSON: `docs/tier0_results.json`,
`docs/phase2_mask_cost.json`, `docs/phase2_window_cost*.json`, `docs/phase25_tier1*.json`.*
