# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Note: a `CLAUDE.md` exists in the parent directory (`~/Documents/GitHub/CLAUDE.md`) covering Nx workspace conventions. It does not apply to this repository — there is no Nx, Node package, or JS build here.

## What this repository is

A mirror of the `MiniMaxAI/MiniMax-H3` Hugging Face model repository: model configs, tokenizers, remote-code VAE implementations, reproduction scripts, and agent skills. It is **not** an installable package — there is no `setup.py`/`pyproject.toml`, no test suite, and no lint/CI config. `*.safetensors` is gitignored, so no weights are present locally; anything that actually runs the model needs `hf download MiniMaxAI/MiniMax-H3` or a diffusers `from_pretrained` fetch.

H3 generates video with native 32 kHz stereo audio (4–15 s, 24 FPS, 768p by default, 2K via regeneration).

`docs/` and `scripts/scd/` are **local additions, not upstream mirror content** — research for a Separable Causal Diffusion port of H3. `docs/MINIMAX_H3_SCD_PORT_DESIGN.md` is the live design doc; `scripts/scd/tier0_bench.py` is a weights-free microbenchmark whose results are pasted into §2.2.1 and stored in `docs/tier0_results.json`. Do not push these upstream.

## Two checkpoint layouts live side by side

Both describe the same model; do not mix their class names or diffusers versions.

| | Original checkpoint (`FL2VA/`, `Ref2VA/`) | Diffusers modular (repo root) |
|---|---|---|
| Entry | `FL2VA/model_index.json`, `Ref2VA/model_index.json` | `model_index.json`, `modular_model_index.json` |
| Pipeline | `MiniMaxH3Pipeline` (diffusers 0.32.2) | `MiniMaxH3ModularPipeline` (0.36.0.dev0) |
| Classes | `MiniMaxH3DiTModel`, `MiniMaxH3VideoVAE`, `MiniMaxH3AudioVAE`, `MiniMaxH3Qwen3VLHFEncoder` | `MiniMaxH3Transformer3DModel`, `AutoencoderKLMiniMaxH3`, `AutoencoderKLMiniMaxH3Audio`, `MiniMaxH3Scheduler` |
| Consumers | SGLang, vLLM | diffusers (`minimax-h3` branch) |
| Python source | yes (`video_vae/`, `audio_vae/`) | none — code comes from diffusers |

The `_minimax_h3` block in each `FL2VA|Ref2VA/model_index.json` carries the runtime-relevant partition metadata: `partition` (`fl2va`/`ref2va`), supported `tasks` (`t2va`+`fl2va` vs `ref2va`), and `sigma_shift_scales` for video vs audio.

Root-level `transformer/` vs `transformer_ref/` are the FL2VA and Ref2VA Omni-Transformers in modular form; `scheduler/` vs `audio_scheduler/` are the video and audio schedulers.

## `FL2VA/` and `Ref2VA/` are byte-identical except `model_index.json`

Every `.py`, `.json`, and `.yaml` under `FL2VA/video_vae/`, `FL2VA/audio_vae/` is duplicated verbatim in `Ref2VA/`. **Any code fix must be applied to both copies** (see commit `c22aafa` for precedent). Verify with:

```bash
diff -rq FL2VA Ref2VA    # should report only model_index.json
```

## Remote-code loading constraints

Both VAEs load through `config.json:auto_map` with `trust_remote_code=True`.

- `video_vae/minimax_h3_video_vae.py` contains a "dependency manifest" block of `from .x import Y  # noqa: F401` imports. These are load-bearing: diffusers' dynamic-module loader only copies **one** level of relative imports into its cache, so every sibling module must be named there. Adding a new file to `video_vae/` requires adding it to that manifest.
- `audio_vae/` weights and hyperparameters come from sibling `config.yaml` / `metadata.json` referenced by `source_config_path` / `source_metadata_path` in `config.json`, not from `config.json` alone.
- The `dac_*.py` files are the DAC/BigVGAN-derived audio codec. BigVGAN wraps every `Conv1d` in `torch.nn.utils.parametrizations.weight_norm`, so `m.weight.data` assignment is a silent no-op — initialize `weight_v`/`weight_g` instead.

Architecture facts that affect shape math: visual VAE is f16t4d24 (16× spatial, 4× temporal, 24 latent channels) further patchified `1×2×2`, giving 32× effective spatial downsampling into the transformer. Audio VAE encodes each stereo channel independently at 32 kHz → 40 Hz latents, 32 channels. Text encoder is Qwen3-VL-32B, hidden states taken from layer 50; the repo's tokenizer config adds special tokens such as `<d>` (dialogue) and must be used as-is.

## Running the model

Serve locally with SGLang (weights come from the Hub, not this checkout). The README convention is port 30010 for FL2VA and 30011 for Ref2VA:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --host 0.0.0.0 --port 30010 --model-variant fl2va
```

`scripts/readme/*.sh` are the reproduction scripts referenced from the README tables; they require `curl` and `jq`.

- `reproducible-768p-{t2va,fl2va,ref2va}-request.sh` — hit a local deployment: `POST /v1/videos` (with `task`, `prompt`, `conditions`, `target.{short_edge,aspect_ratio,duration_seconds}`, `seed`), then `GET /v1/videos/{id}` for status and `GET /v1/videos/{id}/content` to download.
- `full-2k-*.sh` — the three-stage 2K pipeline against the hosted API. Requires `MINIMAX_API_BASE` (`https://api.minimaxi.com` CN / `https://api.minimax.io` global) and `TOKEN`. Stages: `POST /v2/h3_context_ir` → local H3-Base → `POST /v2/video_regeneration`, each polled via `GET /v2/query/video_generation/{task_id}`. The context-IR stage exports `EXPANDED_PROMPT`, which later stages consume, so these scripts are meant to be sourced in order within one shell.

H3-Context-IR (prompt expansion) and H3-Regenerate-2K are hosted-only and not open-sourced; only H3-Base runs locally.

## Prompt structure

Prompts fed to H3-Base are the expanded Context-IR representation, not free-form text. Base modes (T2VA/I2VA/FL2VA/L2VA) use the sections `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` in that order. Ref2VA uses `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`. Reference labels (`<Subject 1>`, `<Video 1>`, `<Audio 1>`) must stay consistent across sections, and dialogue is wrapped in `<d>[Language] ...</d>`. Full rules and examples live in `skills/h3-prompt-writing/references/base-en.txt` and `ref-en.txt`; use the `h3-prompt-writing` skill when writing or rewriting prompts.

## Skills directory

`skills/` is the source of truth: `h3-prompt-writing` (portable — plain Markdown plus reference files, no external calls) and eight style-specific generators that target the MiniMax Hub canvas workflow (`hub_generate_video`, `hub_generate_image`, choice cards) and are **not** portable to generic agent harnesses.

`h3-prompt-writing` is checked in three times — `skills/`, `.claude/skills/`, `.agents/skills/` — as byte-identical copies produced by `npx skills add`. Edit `skills/` and mirror to the other two; `skills-lock.json` stores a SHA-256 of `skills/h3-prompt-writing/SKILL.md` and goes stale if only some copies change.

Style skills carry a `meta.yaml` (bilingual display name, version, tags, descriptions) alongside `SKILL.md` and `SKILL.cn.md`; keep the English and Chinese variants in sync when editing either. When adding or renaming a skill, update both `README.md` and `skills/README.md`, whose links are keyed to the folder name (see commit `5970421` for a mismatch this caused).

The four READMEs (`README.md`, `README.zh-CN.md`, `README.ko.md`, `README.ja.md`) are translations of the same document — substantive changes belong in all four.
