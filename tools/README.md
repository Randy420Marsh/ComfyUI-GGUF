# ComfyUI-GGUF — conversion & quantization tools

This directory contains the standalone Python tools used to convert
diffusion-model `.safetensors` checkpoints into the GGUF format that the
ComfyUI-GGUF custom node loads, plus the patched `llama-quantize` build
recipe used to compress them further.

The tools here are **separate** from the inference-time custom node: they
have their own dependencies (see [`requirements-conversion.txt`](requirements-conversion.txt))
and are intended to be run from their own Python virtualenv, not from
ComfyUI's environment.

For a step-by-step walkthrough that covers a full end-to-end conversion of
a specific model (Z-Image / Lumina2 / Flux / ERNIE-Image), see
[`docs/CONVERSION_GUIDE.md`](../docs/CONVERSION_GUIDE.md). The same content
is also browseable on the wiki at
[`Conversion-Guide`](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki/Conversion-Guide),
which links to a separate
[`Build the patched llama-quantize`](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki/Build-llama-quantize)
page using the pre-patched
[`Randy420Marsh/llama.cpp` `city96` branch](https://github.com/Randy420Marsh/llama.cpp/tree/city96)
(no `git apply` step required).

---

## Files in this directory

| File | Purpose |
|---|---|
| `convert.py` | Convert a `.safetensors` UNET / DiT into a `BF16` or `F16` GGUF. |
| `gguf_gui.py` | Qt GUI that wraps `convert.py` + `llama-quantize` + the inspect/fix helpers, with GPU auto-detection, dtype mode, quant selector, and an Analyze button. |
| `analyze_model.py` | Library used by the Analyze button (also runnable from the CLI). Reads a `.safetensors` header **or** a `.gguf` tensor index and produces a per-quant VRAM table — see [Analyze](#analyze-pick-a-quant-from-the-model-and-your-gpu). |
| `inspect_gguf.py` | Print arch / file type / per-tensor dtype histogram of an existing GGUF. Has `--check-no-bf16` for CI / scripting, `--check-sizes` for catching zero-byte / corrupt tensors that crash `llama-quantize` with `basic_ios::clear: iostream error`, and `--metadata` for the full KV section. |
| `fix_pad.py` | Reshape 1-D `x_pad_token` / `cap_pad_token` to `[1, dim]` on Z-Image / Lumina2 GGUFs so `llama-quantize` doesn't choke on them. Auto-detects no-op cases (Z-Image 0.36 non-Turbo and similar checkpoints already ship 2-D pad tokens) and fast-copies the file unchanged in those cases. **Run between `convert.py` and `llama-quantize` for those models, not after.** |
| `fix_5d_tensors.py` | Re-attach the 5D tensors that `convert.py` strips from Hunyuan Video / Wan 2.1 GGUFs. Run **after** `llama-quantize`. |
| `read_tensors.py` | Dump tensor shapes / dtypes from a `.safetensors` file. Debugging helper. |
| `fix_lines_ending.py` | One-shot CRLF → LF conversion of `lcpp.patch` for users whose Git rewrote line endings on clone. |
| `lcpp.patch` | The patch applied to upstream `llama.cpp` tag `b3962` to add image-model quant math. |
| `requirements-conversion.txt` | Minimal Python deps for the conversion pipeline. |

---

## Prerequisites

- Python ≥ 3.10 (3.11 recommended; ≥ 3.10 is what the GUI was tested on)
- A C++17 compiler (GCC ≥ 12, Clang ≥ 15, or MSVC 2019/2022)
- CMake ≥ 3.21
- Git
- (Optional, GPU quantize) NVIDIA CUDA toolkit ≥ 12.0. CPU-only `llama-quantize` is fine for most diffusion models and is what the build below uses by default. Pass `-DGGML_CUDA=ON` to `cmake` only if you specifically want CUDA-accelerated quant kernels.

### Linux

```bash
sudo apt update
sudo apt install build-essential cmake git
```

### Windows

Install Visual Studio 2019 or 2022 with the **Desktop development with C++** workload (provides MSVC + CMake + Git). See [Platform notes](#platform-notes) for the VS-version-specific cmake flag.

### macOS

```bash
xcode-select --install
brew install cmake
```

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/Randy420Marsh/ComfyUI-GGUF.git
cd ComfyUI-GGUF
```

The conversion tooling, the Qt GUI front-end (`gguf_gui.py`), `requirements-conversion.txt`, `inspect_gguf.py --metadata`, and the long-form `docs/CONVERSION_GUIDE.md` all live in this fork. The upstream [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) repo has the original `convert.py` but does **not** carry any of these additions, so cloning that one will leave you without the GUI and the architectures this README describes (Ministral-3/ERNIE-Image, Gemma-4 tokenizer.json sidecar, scaled-fp8 dequant, etc.).

### 2. Create a conversion virtualenv and install dependencies

The recommended toolchain is [`uv`](https://docs.astral.sh/uv/) — `uv venv` + `uv pip install` is much faster than `python -m venv` + `pip install`, and `uv` picks the right interpreter on its own. Install it once if you haven't already (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
# Linux / macOS (uv, recommended)
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -r tools/requirements-conversion.txt
```

```bat
:: Windows (uv, recommended)
uv venv venv --python 3.11
venv\Scripts\activate.bat
uv pip install -r tools\requirements-conversion.txt
```

Plain `pip` works fine too if you'd rather not install `uv`:

```bash
# Linux / macOS (pip)
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r tools/requirements-conversion.txt
```

```bat
:: Windows (pip)
python -m venv venv
venv\Scripts\activate.bat
pip install -U pip
pip install -r tools\requirements-conversion.txt
```

The `requirements-conversion.txt` file lists minimum versions; newer is fine. `torch` is required even on CPU-only conversion machines because `convert.py` uses it for the actual tensor work; install whichever build matches your platform (the CPU wheel from PyPI is sufficient for conversion). A recently validated combination is `Python 3.11 + uv + torch 2.12 + safetensors 0.7 + numpy 2.4 + gguf 0.19 + PySide6 6.11` on Ubuntu with CUDA 13; older floors all the way back to torch 2.1 / numpy 1.24 still work.

> **Why a separate venv?** The root-level `requirements.txt` / `pyproject.toml` only list what the **custom node** needs at inference time (`gguf` + tokenizer extras). Those install into ComfyUI's environment. The conversion tools have a different dependency surface (torch, safetensors, gguf, Qt for the GUI) and shouldn't pollute the inference env.

### 3. Build the patched `llama-quantize`

Standard upstream `llama.cpp` doesn't know how to quantize image-model tensors (the K/IQ block kernels were written for LLM weight shapes). The `lcpp.patch` here teaches it to.

> **Where to clone:** The GUI and the documented CLI examples both assume `llama.cpp` lives **inside the `ComfyUI-GGUF` repo root**, i.e. you'll end up with:
>
> ```
> ComfyUI-GGUF/
> ├── llama.cpp/
> │   └── build/
> │       └── bin/
> │           └── llama-quantize     <-- the GUI looks for this exact path
> ├── tools/
> ├── loader.py
> └── …
> ```
>
> All the `git clone` commands below assume your current working directory is the `ComfyUI-GGUF` repo root. If you keep `llama.cpp` somewhere else, point the GUI at it with `export LLAMA_CPP_DIR=/abs/path/to/llama.cpp` (see [Setup step 4](#4-linux-only-export-ld_library_path) for the CLI equivalent). The GUI also runs a pre-flight existence check before Step 1 so you don't waste 30+ seconds and 12+ GiB writing an intermediate F16 GGUF only to fail because the binary is missing.

**Shortcut (recommended):** from the `ComfyUI-GGUF` repo root, clone the pre-patched [`city96` branch](https://github.com/Randy420Marsh/llama.cpp/tree/city96) of [`Randy420Marsh/llama.cpp`](https://github.com/Randy420Marsh/llama.cpp). That branch is upstream `ggml-org/llama.cpp` at tag `b3962` with `lcpp.patch` already applied — no `git apply` step, no CRLF normalisation, no `--ignore-whitespace` workaround:

```bash
cd /path/to/ComfyUI-GGUF        # <-- repo root, NOT tools/
git clone -b city96 https://github.com/Randy420Marsh/llama.cpp.git
cd llama.cpp
```

The longer wiki page [Build the patched llama-quantize](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki/Build-llama-quantize) covers CUDA-build variants and the smoke-test step in more detail.

<details>
<summary><b>Manual patch path (if you'd rather not trust the fork or want a different upstream base)</b></summary>

From the `ComfyUI-GGUF` repo root, clone `llama.cpp` **into** the repo (not next to it), check out the exact tag the patch is written against, and apply it:

```bash
cd /path/to/ComfyUI-GGUF        # <-- repo root, NOT tools/
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout tags/b3962
git apply ../tools/lcpp.patch
```

If `git apply` complains about line endings, run `python ../ComfyUI-GGUF/tools/fix_lines_ending.py` first (it converts `lcpp.patch` CRLF → LF in place) and retry. As a last resort, `git apply --ignore-whitespace ../ComfyUI-GGUF/tools/lcpp.patch` also works.

</details>

Build just the quantizer target (don't build everything — `llama.cpp` is huge):

```bash
mkdir build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON
cmake --build build --config Release -j$(nproc) --target llama-quantize
cd ..
```

The `-DCMAKE_CXX_STANDARD=17` flag is the important one — modern CUDA toolkits (12.x and 13.x) **require** C++17, and CMake will default to a lower standard on some platforms if you don't pin it. See [Platform notes](#platform-notes) for VS2022 and CUDA-build variants.

After the build, the binary lives at:

- Linux / macOS: `<ComfyUI-GGUF>/llama.cpp/build/bin/llama-quantize`
- Windows: `<ComfyUI-GGUF>\llama.cpp\build\bin\Release\llama-quantize.exe` (or `Debug\` if you used `--config Debug`)

Verify before launching the GUI:

```bash
# from the ComfyUI-GGUF repo root
ls -l llama.cpp/build/bin/llama-quantize          # Linux / macOS
dir llama.cpp\build\bin\Release\llama-quantize.exe :: Windows
```

If this path doesn't exist, the GUI will refuse to start Step 1 with a clear pre-flight error pointing you back here.

### 4. (Linux only) Export `LD_LIBRARY_PATH` (CLI usage only)

The patched `llama-quantize` build links against the shared `ggml` libraries inside the `llama.cpp/build/` tree, not against any system-installed `libggml`. If you run it from a different working directory and the runtime loader can't find those `.so` files, the binary will fail with `error while loading shared libraries: libggml.so: cannot open shared object file`.

**This step is only required for direct CLI invocations of `llama-quantize`.** The GUI (`gguf_gui.py`) computes `LD_LIBRARY_PATH` automatically from `LLAMA_CPP_DIR` (default: `<repo>/llama.cpp`) for the conversion subprocess, so you don't need to export anything for the GUI workflow.

For CLI usage, export the path before invoking `llama-quantize`:

```bash
export LD_LIBRARY_PATH=./llama.cpp/build/src:./llama.cpp/build/ggml/src:$LD_LIBRARY_PATH
```

Make this persistent by appending the line to your `~/.bashrc` / `~/.zshrc`. The `./` prefixes assume your working directory is the root of this repo when you run the conversion — adjust to absolute paths if you call it from elsewhere (e.g. `export LD_LIBRARY_PATH=/path/to/ComfyUI-GGUF/llama.cpp/build/src:/path/to/ComfyUI-GGUF/llama.cpp/build/ggml/src:$LD_LIBRARY_PATH`).

macOS uses `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`. Windows doesn't need any equivalent — the loader picks up DLLs from `llama.cpp\build\bin\Release\` automatically.

---

## Usage

There are two ways to drive the pipeline: the GUI (recommended for most users) and the CLI (recommended for scripting / CI).

### Option A — GUI (`gguf_gui.py`)

```bash
source venv/bin/activate
python tools/gguf_gui.py
```

The GUI resolves `convert.py`, `fix_pad.py`, and the patched `llama-quantize` binary from its own location (and from `<repo>/llama.cpp/build/`), so you can launch it from any working directory — either from the repo root as shown above, or directly from inside `tools/`:

```bash
cd tools
python gguf_gui.py
```

If your `llama.cpp` clone lives somewhere other than `<repo>/llama.cpp`, point the GUI at it explicitly:

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp
python tools/gguf_gui.py
```

`LD_LIBRARY_PATH` is set automatically based on `LLAMA_CPP_DIR`, so step 4 of [Setup](#4-linux-only-export-ld_library_path) is only needed when running `llama-quantize` directly on the CLI.

The GUI handles convert → fix → quantize → inspect end-to-end. Key controls:

- **Source / Destination** — input `.safetensors` and output directory.
- **Dtype mode** — `Auto (detect via nvidia-smi)` / `Force F16` / `Force BF16`.
  - `Auto` queries `nvidia-smi --query-gpu=name,compute_cap`, finds the lowest compute capability across visible GPUs, and picks BF16 only for Ampere+ (CC ≥ 8.0). Turing (CC 7.5, e.g. RTX 20xx) and earlier get F16. **This is the right default for RTX 20-series users** — bf16 weights on Turing get up-cast to fp32 at runtime, doubling memory and killing throughput.
  - `Force F16` / `Force BF16` are debug overrides.
  - A status label below the combo shows the detected GPU + resolved `--dtype` flag.
- **Quantization type** — full list of `llama-quantize` output types (`F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q4_K_S`, `IQ3_M`, `Q3_K_M`, `Q3_K_S`, `IQ2_M`, `Q2_K`, …). Default `Q4_K_M`. Saved between runs in `settings.json`.
- **Analyze** — see below.
- **Convert** — runs the full pipeline. Output appears in a log pane.

#### Analyze (pick a quant from the model and your GPU)

Click **Analyze** next to **Browse** to read the model's tensor index (no weight load) and pop a dialog with a per-quant VRAM table. Works on both `.safetensors` and `.gguf` inputs — the former parses the JSON header at offset 8, the latter walks the GGUF tensor list via `gguf.GGUFReader`. Useful when re-evaluating an already-converted intermediate (e.g. comparing the city96 pre-quantized Z-Image Turbo against a fresh Z-Image 0.36 F16 conversion). The estimate is **100% model-derived** — no hardcoded "Flux is 12B" tables. It:

1. Detects the architecture by re-using the same `ModelXxx.keys_detect` logic `convert.py` uses (ERNIE, Flux, Lumina2/Z-Image, SD3, Hunyuan, SDXL, SD1, …), so detection is authoritative.
2. Computes **weight bytes per quant** by mirroring `convert.py`'s promote-to-F32 rules (1-D tensors / ≤ 1024 elems / `keys_hiprec` blacklist stay F32).
3. Computes **activation budget** from model-derived dims (hidden_dim from a known reference key, layer count from prefix scan, patch_size from arch) with the **VAE 8x downsample applied before patchification** (every arch in `IMG_ARCH_LIST` runs on 1/8 latents, not pixels).
4. Cross-references the detected GPU's VRAM (via `nvidia-smi`) with a 1 GB headroom buffer + 400 MB ComfyUI overhead. The highest-quality quant that fits at 1536×1024 is highlighted as the recommendation; a "Use this quant" button propagates the choice to the main quant combo.

The dialog includes the exact formula in monospace so estimates can be audited:

```
weight = sum_tensors(n_params * bpw[quant] / 8, F32 if 1D / <=1024 elems / keys_hiprec)
latent_seq = (W/8/patch) * (H/8/patch)   (8x VAE downsample applied before patchification)
activations = latent_seq * hidden_dim * 3 * 2 bytes   (SDPA / flash attention; ±25% on Turing with --lowvram)
total = weight + activations + 400 MB ComfyUI overhead
fits = total <= (VRAM - 1 GB headroom)
```

The budget intentionally **excludes** the VAE and text encoder — those are loaded by separate `*CLIPLoader (gguf)` / `Load VAE` nodes in the standard ComfyUI-GGUF workflow and don't co-reside with the diffusion model during the denoising step.

### Option B — CLI

#### Convert `.safetensors` → GGUF (intermediate F16 / BF16)

```bash
python tools/convert.py --src /path/to/model.safetensors
```

Output filename defaults to `<src-stem>-{BF16|F16}.gguf` next to the source. Override with `--dst /path/to/out.gguf`.

Flags:

- `--dtype {auto,fp16,bf16}` (default `auto`)
  - `auto` — bf16-source weights stay BF16 in the GGUF; fp16-source stays F16. Preserves the source precision.
  - `fp16` — every bf16-source weight is cast to fp16 before going into the GGUF, and the file-level type is `MOSTLY_F16`. **Use this on Turing (RTX 20xx) or anywhere bf16 isn't natively supported.**
  - `bf16` — explicit override; same as `auto` when the source is bf16, but won't down-cast fp16 sources.

> **Note**: do not use the diffusers UNET format for Flux, it won't work — use the default reference checkpoint key format. Q/K/V are merged into one `qkv` key in the diffusers format and the converter can't split them. Load the diffusers checkpoint in ComfyUI and save it with the built-in `ModelSave` node to get the reference layout.

> **Hunyuan Video / Wan 2.1**: you'll see a warning about 5D tensors. The converter saves a **non-functional** intermediate GGUF (so it can be passed to `llama-quantize`). After quantization, run `fix_5d_tensors.py` to re-attach the missing 5D tensor — see [`fix_5d_tensors.py` usage](#fix-5d-tensors-hunyuan-video--wan-21).

#### Quantize with the patched `llama-quantize`

```bash
./llama.cpp/build/bin/llama-quantize \
  /path/to/model-F16.gguf \
  /path/to/model-Q4_K_M.gguf \
  Q4_K_M
```

Run `./llama.cpp/build/bin/llama-quantize --help` for the full list of output types and their per-block bit-rates. The most common picks for diffusion models on consumer GPUs:

| Quant | Bits per weight | Typical use |
|---|---:|---|
| `F16` | 16.0 | No compression, fp16 reference |
| `Q8_0` | 8.5 | Near-lossless; fits 6-7B models on 12 GB with text-encoder offload |
| `Q6_K` | 6.14 | Slight quality drop from F16; good on 12-16 GB VRAM |
| `Q5_K_M` | 5.5 | Sweet spot for 6-7B models on 8 GB (tight) / 12 GB (comfortable) |
| `Q4_K_M` | 4.58 | **Default**. Fits 12B Flux on 8 GB. Small quality loss. |
| `Q4_K_S` | 4.36 | Slightly smaller than `Q4_K_M`, slightly more quality loss |
| `Q3_K_M` | 3.41 | Last "still pretty good" tier |
| `IQ2_M` / `Q2_K` | ~2.5 | Aggressive; visible artifacts on most models |

Don't quantize SDXL / SD1 / other Conv2D-heavy models directly — extract the UNET first.

#### Fix padding tokens (Z-Image / Lumina2)

Between `convert.py` and `llama-quantize`, for Z-Image / Lumina2 only:

```bash
python tools/fix_pad.py /path/to/zimage-F16.gguf
# -> writes /path/to/zimage-F16_fixed.gguf
```

Behaviour:

- **1-D pad tokens present** (Z-Image Turbo legacy, RedCraft ZiB) → reshapes them to `[1, dim]` and rewrites the GGUF.
- **Pad tokens already 2-D** (e.g. Z-Image 0.36 non-Turbo from [`OmegaShred/Z-Image-0.36`](https://huggingface.co/OmegaShred/Z-Image-0.36)) → prints `No 1-D pad tokens found -- nothing to fix.` and copies the file unchanged. Safe to leave in the pipeline.

The GUI skips this step entirely when no fix is needed, feeding the F16 GGUF directly to `llama-quantize`.

Skipping this step on a Z-Image Turbo (pre-2-D) model will cause `llama-quantize` to fail or produce a model that generates severe noise.

#### Fix 5D tensors (Hunyuan Video / Wan 2.1)

After `llama-quantize`, for Hunyuan Video / Wan 2.1 only:

```bash
python tools/fix_5d_tensors.py \
  --src /path/to/wan2.1-t2v-1.3b-Q8_0.gguf \
  --dst /path/to/wan2.1-t2v-1.3b-Q8_0-fixed.gguf
```

This also writes a `fix_5d_tensors_<arch>.safetensors` cache file into `tools/`; delete it after all your conversions of that arch are done.

#### Inspect a GGUF

```bash
python tools/inspect_gguf.py /path/to/model.gguf
```

Prints arch + file type + per-tensor dtype histogram. Useful flags:

- `--check-no-bf16` — exits non-zero (rc=2) if any `BF16` tensor sneaked in. Drop this into CI when you specifically need an F16-only GGUF (Turing target).
- `--check-sizes` — exits non-zero (rc=3) if any tensor in the GGUF has zero bytes or a stored size that doesn't match `prod(shape) * dtype-size`. Run this on `_f16.gguf` (and `_f16_fixed.gguf`) when `llama-quantize` crashes mid-tensor with `basic_ios::clear: iostream error`; the offending tensor is the one in the failure log with `size = 0.000 MB`.
- `--verbose` — full per-tensor table (`name`, `shape`, `dtype`).
- `--metadata` — dump the KV section (architecture, attention head counts, context length, tokenizer config, etc.). Use this instead of trying to point `safetensors.safe_open` at a GGUF — GGUF and safetensors are different on-disk formats and pointing the safetensors reader at a GGUF gets you `header too large` errors.

---

## Platform notes

### Windows + Visual Studio 2019

```bat
mkdir build
cmake -B build
cmake --build build --config Release -j10 --target llama-quantize
```

### Windows + Visual Studio 2022

VS2022's MSVC defaults to C++14, which doesn't compile modern CUDA/llama.cpp. Force C++17 explicitly:

```bat
mkdir build
cmake -B build ^
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -DCMAKE_CXX_FLAGS="/std:c++17"
cmake --build build --config Release -j10 --target llama-quantize
```

If `llama.cpp\common\log.cpp` fails to compile with chrono-related warnings, insert two lines after the existing first line of that file:

```cpp
#include "log.h"

#define _SILENCE_CXX23_CHRONO_DEPRECATION_WARNING
#include <chrono>
```

Then re-run the build command.

### CUDA-accelerated quantize

If you specifically want `llama-quantize` to use CUDA kernels (faster on very large models, no real benefit on most 6-12B image models), add `-DGGML_CUDA=ON -DCMAKE_CUDA_STANDARD=17` to the `cmake -B build` step. Requires the NVIDIA CUDA toolkit installed and visible to CMake.

### macOS

Use `DYLD_LIBRARY_PATH` (not `LD_LIBRARY_PATH`):

```bash
export DYLD_LIBRARY_PATH=./llama.cpp/build/src:./llama.cpp/build/ggml/src:$DYLD_LIBRARY_PATH
```

`-DGGML_METAL=ON` enables Metal-backed quant kernels on Apple Silicon. CPU-only is fine for diffusion models.

---

## Troubleshooting

- **`git apply` reports "patch does not apply"**: you're not on `tags/b3962`, or your Git rewrote `lcpp.patch` to CRLF on clone. Verify the tag (`git -C llama.cpp describe --tags`) and re-run with `python tools/fix_lines_ending.py && git -C llama.cpp apply ../ComfyUI-GGUF/tools/lcpp.patch`.
- **`error while loading shared libraries: libggml.so`**: export `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) per the [setup section](#4-linux-only-export-ld_library_path).
- **`safetensors_rust.SafetensorError: Error while deserializing header: header too large`**: you're pointing a safetensors reader at a `.gguf` file. They're different on-disk formats. Use `python tools/inspect_gguf.py --metadata <file>` instead.
- **GUI says "GPU not detected"**: `nvidia-smi` isn't on PATH (CPU-only / ROCm / Apple machine). The GUI still works — Analyze just disables the Fits / recommendation columns. Use `Force F16` as the dtype mode.
- **bf16 model is huge and slow on RTX 20xx**: bf16 weights get up-cast to fp32 at runtime on Turing because it has no native bf16 support. Re-convert with `--dtype fp16` (CLI) or `Auto`/`Force F16` (GUI) to produce an F16 GGUF instead.
- **`Unexpected text model architecture type in GGUF file: 'mistral3'`**: you're on an older `loader.py` than what this repo currently ships. Pull the latest `main` — `mistral3` (Ministral 3B / ERNIE-Image text encoder) is supported.

---

## Contributing back to `lcpp.patch`

If you change something in the patched `llama.cpp` source and want to share it:

```bash
cd llama.cpp
git diff src/llama.cpp > ../ComfyUI-GGUF/tools/lcpp.patch
```

(Adjust the path to `git diff` depending on which subtree you modified.)
