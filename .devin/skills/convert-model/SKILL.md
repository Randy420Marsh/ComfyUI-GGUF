---
name: convert-model
description: Convert a .safetensors diffusion model to quantized GGUF via tools/gguf_pipeline.py
argument-hint: "[src.safetensors] [quant-type]"
allowed-tools:
  - read
  - grep
  - glob
  - exec
---

Convert a `.safetensors` checkpoint to a quantized GGUF using the one-shot
pipeline CLI. Source file: $1 — quant type: $2 (default `Q4_K_M`).

## Environment (MANDATORY)

Use the dedicated conversion venv — never ComfyUI's environment:

```bash
cd <repo-root>
uv venv venv --python 3.11          # only if venv/ does not exist yet
source venv/bin/activate
uv pip install -r tools/requirements-conversion.txt   # only on first setup
```

## Pre-flight checks (do these BEFORE converting)

1. Verify the patched `llama-quantize` binary exists at
   `<repo-root>/llama.cpp/build/bin/llama-quantize` (or at
   `$LLAMA_CPP_DIR/build/bin/llama-quantize` if the env var is set).
   If missing, STOP and tell the user to build it first — see
   `tools/README.md` § 3 or the wiki page "Build-llama-quantize".
   Shortcut: `git clone -b city96 https://github.com/Randy420Marsh/llama.cpp.git`
   then cmake-build the `llama-quantize` target.
2. Verify the source file exists and check free disk space: the pipeline
   writes an intermediate F16/BF16 GGUF roughly the size of the source
   (often 12+ GiB) before quantizing.
3. Optionally recommend a quant first:
   `python tools/analyze_model.py /path/to/model.safetensors`
   (works on `.gguf` inputs too).

## Run the pipeline

```bash
python tools/gguf_pipeline.py \
  --src "$1" \
  --dst-dir <output-dir> \
  --quant ${2:-Q4_K_M}
```

Notes:
- On Turing (RTX 20xx) or older GPUs add `--dtype fp16` (no native BF16).
- The pipeline automatically handles: 5-D tensor pre-fold (Hunyuan Video /
  Wan), the Z-Image / Lumina2 pad-token fix (`fix_pad.py`, auto-skipped when
  not needed), `LD_LIBRARY_PATH` for the llama.cpp shared libs, and cleanup
  of intermediates. Do NOT run `fix_pad.py` / `fix_5d_tensors.py` manually
  when using `gguf_pipeline.py`.
- All 39 supported quant names: see `pipeline_lib.LLAMA_QUANTIZE_TYPES` or
  `python tools/gguf_pipeline.py --help`.

## Validate the output

```bash
python tools/inspect_gguf.py <output>.gguf --check-sizes
```

Report the final output path, file size, and the inspect summary
(arch, file type, dtype histogram). Per-model details and troubleshooting:
`docs/CONVERSION_GUIDE.md`.
