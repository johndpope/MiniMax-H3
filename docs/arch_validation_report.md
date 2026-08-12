# SCD architecture validation report

- **When:** 2026-08-12T01:52:21Z
- **Git:** `ed06a2c`
- **Overall:** **PASS**

This report answers: *is the SCD (Separable Causal Diffusion) split still correct, consistent, and as fast as we measured?* It is not a quality check for decoded video pixels.

## 1. Code contracts (unit tests)

| Suite | Result | Cases | Time |
|-------|--------|-------|------|
| `test_scd_attention.py` | PASS | 19/19 | 6.398s |
| `test_scd_model.py` | PASS | 14/14 | 1.276s |
| `test_scd_lora.py` | PASS | 12/12 | 1.292s |
| `test_phase3_train.py` | PASS | 10/10 | 3.058s |
| `test_phase3_sample.py` | PASS | 6/6 | 3.05s |

## 2. Recorded measurements & claims

- **check_findings:** PASS (0 errors, 7 warnings)
  - WARN: docs/phase0_results.json: superseded by docs/phase0_validation_ondist.json — not a live result
  - WARN: docs/phase0_results.json: latent_t=8 is off the DiT's 5n+2 grid (2, 7, 12, ...)
  - WARN: docs/phase0_results.json: sigmas [0.0, 0.1, 0.25] sit below 0.5, where H3 has ~no training density — superseded run, or off-distribution
  - WARN: docs/phase0_validation.json: superseded by docs/phase0_validation_ondist.json — not a live result
  - WARN: docs/phase25_tier1.json: latent_t 12: window 12 is at least the clip length — the encoder ran unbounded here, so this row does not price the shipped configuration
  - WARN: docs/phase2_window_cost.json: clip sigma 0.5714285714: window 6 only evicts after the final chunk (latent_t=7, chunk_frames=1) — a control for `cache.keep`, not a measurement of what a window costs
  - WARN: docs/phase2_window_cost.json: clip sigma 0.9230769231: window 6 only evicts after the final chunk (latent_t=7, chunk_frames=1) — a control for `cache.keep`, not a measurement of what a window costs

## 3. Headline numbers (from stored benches)

**Tier 0** (weights-free primitives on `NVIDIA RTX PRO 4000 Blackwell`):

| Config | Tokens | Steps (N) | Measured speedup | FLOP-model | Drift |
|--------|--------|-----------|------------------|------------|-------|
| 512^2, T=4 (Phase 3 train) | 1,024 | 16 | **1.41×** | 1.41× | 0.1% |
| 768p, 5s | 31,248 | 16 | **2.67×** | 2.72× | 1.7% |
| 768p, 10s | 61,488 | 16 | **3.90×** | 4.01× | 2.7% |
| 768p, 15s | 91,728 | 16 | **5.03×** | 5.24× | 3.9% |

- Kill criterion (SCD faster at 768p/15s): **yes**

**Tier 1** (real weights, untrained split — encoder depth 30, window 12):

| Length / steps | Speedup |
|----------------|---------|
| latent_t=12_N=8 | **2.93×** |
| latent_t=12_N=16 | **3.51×** |
| latent_t=12_N=30 | **3.88×** |

- Flat KV cache (key-value memory pinned ~constant across length): **yes** (spread 38 rows; rows=[12758, 12778, 12796])
- Decoder time per frame (should stay ~flat): [212.91, 215.84, 214.31] ms (spread 1.4%)

## 4. What this does **not** claim

- Pixel quality / blur (that is Phase 3 sampling + VAE decode).
- Training loss or LoRA rank adequacy.
- That re-running Tier 0 live is required every time (use `--bench` when silicon or dims change).

