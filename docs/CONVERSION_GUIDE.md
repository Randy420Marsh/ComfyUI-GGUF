# Conversion Guide — `.safetensors` → patched-GGUF for ComfyUI-GGUF

This guide is the long-form, per-model walkthrough. It complements
[`tools/README.md`](../tools/README.md), which is the short setup &
reference doc for the conversion tools themselves.

If you've never set up the conversion pipeline before, do
[`tools/README.md` Phases 1–4](../tools/README.md#setup) first
(venv, dependencies, `lcpp.patch` + `b3962`, `LD_LIBRARY_PATH`).
The wiki page [Build the patched llama-quantize](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki/Build-llama-quantize)
covers the shortcut version that uses the pre-patched
[`Randy420Marsh/llama.cpp` `city96` branch](https://github.com/Randy420Marsh/llama.cpp/tree/city96)
(no `git apply` step required, no CRLF normalisation).
This guide assumes you already have a working `llama-quantize` binary
at `llama.cpp/build/bin/llama-quantize` and a `.venv-convert/` venv
with `gguf`, `safetensors`, `torch`, `numpy`, `tqdm`, and optionally
`PySide6` installed.

---

## Contents

1. [Two ways to drive the pipeline (GUI vs CLI)](#two-ways-to-drive-the-pipeline)
2. [Phase A — Convert `.safetensors` to an intermediate F16 / BF16 GGUF](#phase-a--convert-safetensors-to-an-intermediate-f16--bf16-gguf)
3. [Phase B — Fix-ups (model-specific)](#phase-b--fix-ups-model-specific)
4. [Phase C — Quantize with `llama-quantize`](#phase-c--quantize-with-llama-quantize)
5. [Phase D — Validate the output](#phase-d--validate-the-output)
6. [Phase E — Load in ComfyUI](#phase-e--load-in-comfyui)
7. [Per-model recipes](#per-model-recipes)
   - [Flux dev / Flux schnell](#flux-dev--flux-schnell)
   - [Z-Image Turbo / Lumina2 / RedCraft ZiB](#z-image-turbo--lumina2--redcraft-zib)
   - [ERNIE-Image (+ Ministral-3-3B text encoder)](#ernie-image--ministral-3-3b-text-encoder)
   - [SD3 / SD3.5](#sd3--sd35)
   - [Hunyuan Video / Wan 2.1](#hunyuan-video--wan-21)
8. [Picking a quant: how the Analyze button decides](#picking-a-quant-how-the-analyze-button-decides)
9. [Reference: quant types & bits-per-weight](#reference-quant-types--bits-per-weight)
10. [Troubleshooting](#troubleshooting)

---

## Two ways to drive the pipeline

| | GUI (`tools/gguf_gui.py`) | CLI |
|---|---|---|
| Setup steps | Read [`tools/README.md`](../tools/README.md#setup) | Read [`tools/README.md`](../tools/README.md#setup) |
| Per-conversion clicks/keys | ~6 clicks | ~3 commands |
| GPU auto-detect (bf16 vs f16) | Yes (`Auto` dtype mode) | No — pass `--dtype fp16` on Turing |
| Quant picker UI | Yes (dropdown of all 30+ output types) | Pass the type name to `llama-quantize` |
| Model-aware quant recommendation | Yes (**Analyze** button) | Run `python tools/analyze_model.py <path> <vram_gb>` |
| Subprocess for `llama-quantize` | Spawned automatically | You invoke it yourself |
| `LD_LIBRARY_PATH` requirement | Inherits from the launching shell | You export it before running |
| Scriptable / CI-friendly | No | Yes |
| Recommended for | One-off conversions, beginners | Batch jobs, scripting, headless servers |

The two paths produce **identical** output GGUFs — the GUI is a wrapper around exactly the same `convert.py` + `llama-quantize` invocations.

---

## Phase A — Convert `.safetensors` to an intermediate F16 / BF16 GGUF

### GUI

1. Launch: `python tools/gguf_gui.py`
2. Click **Browse** next to **Source** → pick your `.safetensors` file.
3. Set **Destination** if you want it somewhere other than next to the source.
4. Set **Dtype mode**:
   - **Auto (detect via nvidia-smi)** — the right default for almost everyone. Queries your GPU's compute capability and picks BF16 only for Ampere+ (CC ≥ 8.0). RTX 20-series (Turing, CC 7.5) and earlier get F16.
   - **Force F16** — only override if `Auto` misdetects or if you specifically need an F16 GGUF on an Ampere+ machine (e.g. shipping a model for a Turing user).
   - **Force BF16** — debug override.
5. (Optional) Click **Analyze** to see the per-quant VRAM table for this specific model and your specific GPU. See [Picking a quant](#picking-a-quant-how-the-analyze-button-decides). You can pick a quant from the dialog with **Use this quant** or just close it and pick manually in the next step.
6. Pick a **Quantization type** in the dropdown (default `Q4_K_M`).
7. Click **Convert**. The log pane shows the `convert.py` run followed by the `llama-quantize` run.

The GUI persists your dtype / quant selections to `settings.json` between launches.

### CLI

```bash
source .venv-convert/bin/activate
python tools/convert.py --src /path/to/model.safetensors
```

Defaults:

- Output filename = `<src-stem>-{BF16|F16}.gguf` next to the source. Override with `--dst /explicit/path/out.gguf`.
- `--dtype auto`: bf16-source → BF16 GGUF, fp16-source → F16 GGUF.

On Turing (RTX 20-series) or anywhere bf16 isn't natively supported, **explicitly pass `--dtype fp16`**. Without it, the converter will emit a BF16 GGUF that subsequent inference will up-cast to fp32 at runtime, doubling weight memory and crushing throughput:

```bash
python tools/convert.py --src /path/to/model.safetensors --dtype fp16
```

The converter knows how to handle these inputs out of the box:

| Source dtype | Handled by | Notes |
|---|---|---|
| `bfloat16` | `--dtype auto` (default) → BF16 GGUF; `--dtype fp16` → F16 GGUF | bf16-source on Turing → use `--dtype fp16` |
| `float16` | → F16 GGUF regardless of `--dtype` | Most public diffusion models |
| `float32` | → BF16 by default; `--dtype fp16` to force F16 | Rare; some training checkpoints |
| `float8_e4m3fn` (ComfyUI scaled fp8) | Auto-dequantized via the sibling `.weight_scale` tensors | No GGUF format encodes fp8 directly; this dequants to F16/BF16 first |
| `float8_e5m2` (ComfyUI scaled fp8) | Same | Same |

If the script warns about **5D tensors**, that's expected for Hunyuan Video / Wan 2.1 — the resulting intermediate GGUF is intentionally **non-functional** so it can still be passed to `llama-quantize`. Re-attach the 5D tensor with `fix_5d_tensors.py` after Phase C. See the [Hunyuan / Wan recipe](#hunyuan-video--wan-21).

> **Diffusers UNET format**: do **not** point `convert.py` at the diffusers-style Flux UNET. It merges Q/K/V into a single `qkv` key that the converter can't split. Load the diffusers checkpoint in ComfyUI, save it with the built-in **ModelSave** node, then convert the saved file.

---

## Phase B — Fix-ups (model-specific)

Two models in particular need a post-convert / pre-quantize fix-up step:

- **Z-Image Turbo / Lumina2 / RedCraft ZiB** → run `fix_pad.py` between Phase A and Phase C. Skipping this step makes `llama-quantize` either fail outright or produce a model whose outputs are pure noise.
- **Hunyuan Video / Wan 2.1** → no Phase B step; the fix-up (`fix_5d_tensors.py`) happens after Phase C.

For everything else (Flux, ERNIE-Image, SD3/3.5, SDXL extracted UNET, Hunyuan-DiT, etc.) — no Phase B step.

```bash
# Z-Image / Lumina2 only
python tools/fix_pad.py /path/to/zimage-F16.gguf
# -> writes /path/to/zimage-F16_fixed.gguf
```

The GUI does this automatically when it detects the arch is `lumina2`.

---

## Phase C — Quantize with `llama-quantize`

### GUI

The GUI runs `llama-quantize` automatically as part of **Convert**, using the type you picked in the **Quantization type** combo. The log pane shows the exact command and its progress per-tensor.

### CLI

```bash
# Linux runtime requirement — see tools/README.md
export LD_LIBRARY_PATH=./llama.cpp/build/src:./llama.cpp/build/ggml/src:$LD_LIBRARY_PATH

./llama.cpp/build/bin/llama-quantize \
  /path/to/model-F16.gguf \
  /path/to/model-Q4_K_M.gguf \
  Q4_K_M
```

The intermediate `model-F16.gguf` can be deleted after quantization unless you want to keep it around as the reference precision (useful for re-quantizing to a different type without re-running `convert.py`).

If you're on Z-Image / Lumina2, feed the **`_fixed.gguf`** output from Phase B into this step, not the raw `model-F16.gguf`.

If you're on Hunyuan Video / Wan 2.1, the intermediate is functionally broken (per the Phase A warning) but `llama-quantize` doesn't care; quantize it and then run the post-fix in [the Hunyuan recipe](#hunyuan-video--wan-21).

### Common quant choices

| Quant | bits / weight | When |
|---|---:|---|
| `F16` | 16.0 | No compression; reference for re-quantizing later |
| `Q8_0` | 8.5 | Near-lossless. Fits 6–7B models on 12 GB w/ text-encoder offload. Tight on 8 GB. |
| `Q6_K` | 6.14 | Slight quality drop from F16. Comfortable on 12–16 GB. |
| `Q5_K_M` | 5.5 | Sweet spot for 6–7B models on 8 GB (tight) / 12 GB (comfortable). |
| `Q4_K_M` | 4.58 | **Default.** Fits 12B Flux on 8 GB. Small quality loss. |
| `Q4_K_S` | 4.36 | Slightly smaller / slightly more quality loss than `Q4_K_M`. |
| `IQ3_M` | 3.66 | Last "still pretty good" tier. |
| `Q3_K_M` | 3.41 | Aggressive — visible degradation on some prompts. |
| `IQ2_M` / `Q2_K` | ~2.5 | Very aggressive; obvious artifacts. |

See [Reference: quant types & bits-per-weight](#reference-quant-types--bits-per-weight) for the full list.

Run `./llama.cpp/build/bin/llama-quantize --help` for the canonical list output by the binary you just built.

---

## Phase D — Validate the output

Quick sanity-check the file before copying it into ComfyUI:

```bash
python tools/inspect_gguf.py /path/to/model-Q4_K_M.gguf
```

Output should look like:

```
File:         /.../model-Q4_K_M.gguf
Architecture: flux        # or sd3 / lumina2 / ernie / hyvid / wan / ...
File type:    MOSTLY_Q4_K_M
Tensors:      780
Dtype histogram:
  Q4_K        420
  Q6_K        140
  Q5_K         12
  F32         208
```

Useful flags:

- `--check-no-bf16` — exits non-zero (rc=2) if any `BF16` tensor sneaked through. Add this to any pipeline targeting Turing.
- `--metadata` — full KV-section dump (architecture, attention head counts, context length, tokenizer model, etc.). Use this instead of pointing `safetensors.safe_open` at a GGUF — different on-disk format; you'll get `header too large` errors.
- `--verbose` — list every tensor with shape + dtype. Useful when something's wrong.

---

## Phase E — Load in ComfyUI

1. Drop the quantized `.gguf` into `ComfyUI/models/diffusion_models/`.
2. In your workflow, replace `Load Checkpoint` (or `UNETLoader`) with **`Unet Loader (GGUF)`** (from this repo).
3. Wire the auxiliary models normally:
   - **Text encoder(s)** — `CLIPLoader (GGUF)` if you have GGUF text encoders; otherwise the stock `CLIPLoader` / `DualCLIPLoader`.
   - **VAE** — stock `Load VAE`.
4. Run.

For ComfyUI-GGUF specifically, the standard workflow keeps the text encoder, VAE, and diffusion model in **separate** loader nodes and **separate** VRAM lifecycles. ComfyUI runs the text encoder first → caches the conditioning → unloads the text encoder → loads the diffusion model → runs the denoising loop → runs the VAE decode at the end. So the **peak VRAM during the denoising step** is just diffusion weights + diffusion activations + ~400 MB of ComfyUI overhead. That's exactly what the [Analyze button](#picking-a-quant-how-the-analyze-button-decides) models.

---

## Per-model recipes

### Flux dev / Flux schnell

No special fix-up. Source: the reference checkpoint format (not the diffusers format — see Phase A warning).

```bash
# Phase A
python tools/convert.py --src /path/to/flux1-dev.safetensors --dtype fp16
# -> /path/to/flux1-dev-F16.gguf

# Phase C
./llama.cpp/build/bin/llama-quantize \
  /path/to/flux1-dev-F16.gguf \
  /path/to/flux1-dev-Q4_K_M.gguf \
  Q4_K_M

# Phase D
python tools/inspect_gguf.py /path/to/flux1-dev-Q4_K_M.gguf
```

Sizing on an 8 GB 2070S: Flux is ~12B params; `Q4_K_M` is the realistic ceiling (~6.5 GB). `Q5_K_M` (~7.8 GB) won't fit alongside activations.

### Z-Image Turbo / Lumina2 / RedCraft ZiB

**Must run `fix_pad.py` between Phase A and Phase C.** Skipping it makes `llama-quantize` either crash or produce a model that generates pure noise.

```bash
# Phase A
python tools/convert.py \
  --src /path/to/redcraftFeb1126Latest_zibDistilledDX3Lucis-full.safetensors \
  --dst /path/to/zimage_f16.gguf \
  --dtype fp16

# Phase B (REQUIRED for this arch)
python tools/fix_pad.py /path/to/zimage_f16.gguf
# -> /path/to/zimage_f16_fixed.gguf

# Phase C
./llama.cpp/build/bin/llama-quantize \
  /path/to/zimage_f16_fixed.gguf \
  /path/to/zimage_Q4_K_M.gguf \
  Q4_K_M
```

The GUI does Phase B automatically when it detects arch = `lumina2`.

**Inference settings for Z-Image / S3-DiT distilled:**

| Parameter | Value | Note |
|---|---|---|
| Steps | 4 – 8 | Distilled models over-bake quickly. Start low. |
| CFG scale | 1.0 – 1.5 | Keep low to avoid artifacting. |
| Sampler | `euler` | Cleanest baseline for S3-DiT. |
| Scheduler | `sgm_uniform` | Effective for fast convergence. |
| Text encoder | Gemma-2-2B via `DualCLIPLoader` | |
| VAE | Whatever Z-Image variant you're using ships | |

### ERNIE-Image (+ Ministral-3-3B text encoder)

ERNIE-Image needs three GGUFs: the diffusion model, the text encoder, and the VAE.

**Diffusion model** — same recipe as Flux, just point `convert.py` at your ERNIE `.safetensors`:

```bash
python tools/convert.py --src /path/to/ernie-image.safetensors --dtype fp16
./llama.cpp/build/bin/llama-quantize \
  /path/to/ernie-image-F16.gguf \
  /path/to/ernie-image-Q4_K_M.gguf \
  Q4_K_M
```

**Text encoder** — ERNIE-Image's official text encoder is **Ministral-3-3B** per [docs.comfy.org](https://docs.comfy.org/tutorials/image/ernie-image/ernie-image). Two ways to get a GGUF:

1. **Pre-built** — e.g. [`dummy9996/Felldude-Uncensored-Ministral3-3B-GGUF`](https://huggingface.co/dummy9996/Felldude-Uncensored-Ministral3-3B-GGUF) or [`unsloth/Ministral-3-3B-Instruct-2512-GGUF`](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF). Drop `Uncensored-Ministral3-3B-Q8_0.gguf` (or the unsloth quant of your choice) into `ComfyUI/models/text_encoders/` and load it with **`CLIPLoader (GGUF)`** with type `ernie`.
2. **Convert your own** — run `llama.cpp/convert_hf_to_gguf.py` against the upstream [`mistralai/Ministral-3-3B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) safetensors, then `llama-quantize` it like above. The upstream HF model is **multimodal** (`Mistral3ForConditionalGeneration` with a Pixtral vision tower), so the converter will emit both `Ministral-3-3B-Instruct-2512.gguf` (LLM) and `mmproj-*.gguf` (Pixtral vision). For ERNIE-Image text-to-image, you only need the LLM file — `ComfyUI-GGUF` ignores the mmproj sibling for text-only tokenizers.

`loader.py` declares `general.architecture = "mistral3"` GGUFs as supported, reads `attention.head_count` / `head_count_kv` from the GGUF metadata (so Ministral-3B's GQA layout `32 / 8` is handled correctly), and auto-reconstructs the Mistral "tekken" tokenizer from the GGUF's `tokenizer.ggml.tokens` field when the model's vocab dim is 131072 (i.e. the entire Mistral family — Mistral Large, Ministral 3B, etc.).

**VAE** — the official `flux2-vae.safetensors`, loaded with the stock `Load VAE` node. No GGUF conversion needed; VAEs are small enough that quant doesn't help.

### SD3 / SD3.5

Same shape as Flux — no fix-up.

```bash
python tools/convert.py --src /path/to/sd3.5_large.safetensors --dtype fp16
./llama.cpp/build/bin/llama-quantize \
  /path/to/sd3.5_large-F16.gguf \
  /path/to/sd3.5_large-Q5_K_M.gguf \
  Q5_K_M
```

SD3.5 Large is ~8B. `Q5_K_M` (~5.5 bpw → ~5.5 GB) is a good fit for 8 GB VRAM with text-encoder offload. `Q8_0` (~8.5 GB) won't fit alongside activations on 8 GB.

### Hunyuan Video / Wan 2.1

These video models have 5D tensors that `convert.py` strips out and saves separately so that `llama-quantize` (which doesn't understand 5D) can process the rest. The Phase A run **will** emit a warning. The intermediate GGUF is **non-functional** until you reattach the 5D tensor after quantization.

```bash
# Phase A — note the warning
python tools/convert.py --src /path/to/wan2.1-t2v-1.3b.safetensors --dtype fp16
# -> /path/to/wan2.1-t2v-1.3b-F16.gguf  (NON-FUNCTIONAL until Phase C+ fix)
# -> /path/to/tools/fix_5d_tensors_wan.safetensors  (cache file, do NOT delete yet)

# Phase C
./llama.cpp/build/bin/llama-quantize \
  /path/to/wan2.1-t2v-1.3b-F16.gguf \
  /path/to/raw/wan2.1-t2v-1.3b-Q8_0.gguf \
  Q8_0

# Phase C+ — re-attach the 5D tensor (REQUIRED for this arch)
python tools/fix_5d_tensors.py \
  --src /path/to/raw/wan2.1-t2v-1.3b-Q8_0.gguf \
  --dst /path/to/wan2.1-t2v-1.3b-Q8_0.gguf
```

Once you've converted all your Hunyuan / Wan models for a given arch, delete the `tools/fix_5d_tensors_<arch>.safetensors` cache file.

Recommended workflow: keep the unfixed intermediate quantized GGUF in a `raw/` subdir so you don't accidentally feed it into ComfyUI before running the fix.

---

## Picking a quant: how the Analyze button decides

Click **Analyze** in the GUI (or run `python tools/analyze_model.py <path.safetensors> <vram_gb>` from the CLI) to get a per-quant VRAM table for *this specific model* on *your specific GPU*. The estimate is **100% derived from the model** — there is no hardcoded "Flux is 12B" table anywhere in `analyze_model.py`.

### What it reads

1. **Architecture** — by re-using the same `ModelXxx.keys_detect` logic that `convert.py` uses to dispatch the converter. So whatever `convert.py` would pick, Analyze picks the same.
2. **Weight bytes per quant** — by walking the safetensors header and applying `convert.py`'s promote-to-F32 rules tensor-by-tensor (1-D tensors / ≤ 1024 elems / arch-specific `keys_hiprec` blacklist stay F32 regardless of the chosen quant). So the weight column is exact for each candidate quant, not an estimate.
3. **Hidden dim** — from a reference tensor key per arch (e.g. `to_q.weight.shape[0]`, `x_embedder.proj.weight.shape[0]`). The exact key it used is shown under `hidden_dim source key:` so the math is auditable.
4. **Layer count** — from a prefix scan over tensor names (`layers.N.` distinct N).
5. **Patch size** — derived from the detected arch class.
6. **GPU VRAM** — from `nvidia-smi --query-gpu=name,memory.total,compute_cap`. If no GPU is visible the recommendation column is left blank but the per-quant weight + activation columns are still computed.

### The activation formula

The dialog shows the formula in monospace so estimates can be audited:

```
weight       = sum_tensors(n_params * bpw[quant] / 8,
                           F32 if 1D / <=1024 elems / keys_hiprec)
latent_seq   = (W/8/patch) * (H/8/patch)     (8x VAE downsample applied
                                              before patchification)
activations  = latent_seq * hidden_dim * 3 * 2 bytes
                                              (SDPA / flash attention;
                                               ±25% on Turing with --lowvram)
total        = weight + activations + 400 MB ComfyUI overhead
fits         = total <= (VRAM - 1 GB headroom)
```

The `W/8` / `H/8` term is the **VAE 8x downsample**. Every diffusion arch in `IMG_ARCH_LIST` (ERNIE, Flux, Lumina2/Z-Image, SD3, SD1, SDXL, Hunyuan-DiT, …) runs on 1/8 latent resolution before patchification — at 1024×1024 px input the transformer actually sees `(1024 / 8 / 2)² = 4096` tokens, not `(1024 / 2)² = 262144`. Without that factor the activation budget would be ~64× too large and Analyze would recommend Q2 for everything.

The `+ 400 MB` is the (measured) overhead of ComfyUI itself once a workflow is loaded and the diffusion model is resident. The `- 1 GB headroom` keeps you out of the "driver shared memory" fallback path on NVIDIA — that path is silent (no OOM) but ~5-10× slower than VRAM-resident execution, which is exactly what you don't want.

### Recommendation logic

The recommendation is the **highest-quality** quant whose total (weight + activations + overhead) fits in `(VRAM - 1 GB)` **at the largest configured resolution** (`1536×1024` by default). The candidate list is sorted by descending bits-per-weight so a higher-quality option is never silently skipped because the loop short-circuited on a smaller-but-still-fitting one.

### What it explicitly does NOT model

- **Text encoder** — separately loaded by `CLIPLoader (GGUF)` / `DualCLIPLoader`, runs first, gets unloaded before the diffusion model loads. Not co-resident at peak. (Verified against the standard ComfyUI-GGUF workflow.)
- **VAE** — runs at the end (decode only; encode also for img2img). Not co-resident with the diffusion model.
- **Sampler step count / CFG path** — steps affect total runtime, not peak memory.
- **KV cache** — diffusion models don't have one; this is an LLM concept.

If your workflow runs the text encoder, VAE, and diffusion model **all resident at once** (monolithic safetensors with embedded encoders, no offload), Analyze underestimates by exactly the sum of those weights. Pick a smaller quant or enable offload.

---

## Reference: quant types & bits-per-weight

The patched `b3962` `llama-quantize` supports the following output types (run `./llama.cpp/build/bin/llama-quantize --help` for the canonical list output by your specific build):

| Type | Bits / weight | Notes |
|---|---:|---|
| `F32` | 32.0 | Uncompressed full precision |
| `F16` | 16.0 | Half precision, no compression beyond fp16 |
| `BF16` | 16.0 | bfloat16; **don't pick for Turing** — no native support |
| `Q8_0` | 8.5 | int8 + per-block fp16 scale. Closest 8-bit option to scaled-fp8. |
| `Q6_K` | 6.14 | |
| `Q5_K_M` | 5.5 | |
| `Q5_K_S` | 5.34 | |
| `Q5_1` | 5.5 | Legacy; prefer `Q5_K_M` |
| `Q5_0` | 5.5 | Legacy; prefer `Q5_K_S` |
| `IQ4_NL` | 4.5 | |
| `IQ4_XS` | 4.25 | |
| `Q4_K_M` | 4.58 | **Default for diffusion models** |
| `Q4_K_S` | 4.36 | |
| `Q4_1` | 4.5 | Legacy; prefer `Q4_K_S` |
| `Q4_0` | 4.5 | Legacy; prefer `Q4_K_S` |
| `IQ3_M` | 3.66 | |
| `IQ3_S` | 3.5 | |
| `IQ3_XS` | 3.3 | |
| `IQ3_XXS` | 3.06 | |
| `Q3_K_L` | 3.5 | |
| `Q3_K_M` | 3.41 | |
| `Q3_K_S` | 3.06 | |
| `IQ2_M` | 2.7 | |
| `IQ2_S` | 2.5 | |
| `IQ2_XS` | 2.31 | |
| `IQ2_XXS` | 2.06 | |
| `Q2_K` | 2.625 | |
| `Q2_K_S` | 2.06 | |
| `IQ1_M` | 1.75 | Very aggressive |
| `IQ1_S` | 1.56 | Very aggressive |
| `TQ1_0` | 1.69 | Ternary; experimental |
| `TQ2_0` | 2.06 | Ternary; experimental |
| `Q4_0_4_4` / `Q4_0_4_8` / `Q4_0_8_8` | 4.5 | ARM-only; do not pick on x86 |
| `COPY` | source | Re-pack without quantizing |

> **No FP8 output type exists in `llama-quantize`.** FP8 in ComfyUI is a runtime concept (`torch.float8_e4m3fn` / `_e5m2` storage type + a `.weight_scale` sibling tensor); `convert.py` already dequantizes those on the way in, but no GGUF on-disk format encodes weights as fp8. The closest 8-bit output is `Q8_0`.

---

## Troubleshooting

### `Unexpected text model architecture type in GGUF file: 'mistral3'`

You're on an older `loader.py` than what this repo currently ships. Pull latest `main` — `mistral3` is supported (since the [PR #5 / PR #6 merges](https://github.com/Randy420Marsh/ComfyUI-GGUF/pull/5)).

### `TypeError: the JSON object must be str, bytes or bytearray, not NoneType` in `load_mistral_tokenizer`

Same — pull latest `main`. The tekken tokenizer reconstruction is now keyed on vocab dim `131072` rather than the hidden dim, which covers both Mistral-Large and Ministral-3B / ERNIE-Image.

### `safetensors_rust.SafetensorError: Error while deserializing header: header too large`

You pointed a safetensors reader at a `.gguf` file. They are completely different on-disk formats — the safetensors loader reads the first 8 bytes as a JSON header length, gets a huge bogus number from the GGUF magic, and bails. Use `python tools/inspect_gguf.py --metadata <file>` instead.

### `error while loading shared libraries: libggml.so: cannot open shared object file`

You forgot the `LD_LIBRARY_PATH` export on Linux. See [`tools/README.md`](../tools/README.md#4-linux-only-export-ld_library_path). On macOS the equivalent variable is `DYLD_LIBRARY_PATH`.

### `git apply: patch does not apply` when applying `lcpp.patch`

Either you're not on `tags/b3962` (`git -C llama.cpp describe --tags` should print `b3962`) or your Git rewrote `lcpp.patch` to CRLF on clone. Run `python tools/fix_lines_ending.py` and retry, or pass `--ignore-whitespace` to `git apply` as a last resort.

### bf16 weights are huge and slow on RTX 20xx (Turing)

Turing has no native bf16 support — every bf16 read is up-cast to fp32 at runtime, doubling weight memory. Re-convert with `--dtype fp16` (CLI) or with **Auto** / **Force F16** in the GUI dtype combo. If you want a CI gate to enforce no bf16, run `python tools/inspect_gguf.py --check-no-bf16 <file>` on the output.

### `5D tensor warning` from `convert.py`

Expected on Hunyuan Video / Wan 2.1. See [the Hunyuan / Wan recipe](#hunyuan-video--wan-21) — the intermediate GGUF is intentionally non-functional and `fix_5d_tensors.py` reattaches the missing tensor after Phase C.

### Analyze recommends `Q2_K` for every model on an 8 GB GPU

Two possibilities:

1. You're analyzing a model with a hidden dim that the arch class doesn't have a reference key for, so Analyze falls back to a large default (2048). Open `analyze_model.py` and check the `hidden_dim source key:` line in the dialog — if it says `(default)` rather than a real key name, the arch detection needs a `keys_hiprec` / hidden-key entry.
2. You're on a machine without `nvidia-smi` and Analyze couldn't read your VRAM, so the recommendation column was left blank. Pass an explicit VRAM in GB as the second CLI arg (`python tools/analyze_model.py <file> 8`) — the GUI doesn't currently have a manual VRAM override.

If neither, please file an issue with the output of `python tools/inspect_gguf.py --metadata <file>` plus `python tools/analyze_model.py <file> <vram_gb>`.
