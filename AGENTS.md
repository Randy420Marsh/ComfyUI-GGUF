# AGENTS.md — ComfyUI-GGUF (Randy420Marsh fork)

Project knowledge for AI coding agents working in this repository.

## What this repo is

GGUF quantization support for native ComfyUI models. Two distinct halves:

1. **Inference custom node** (repo root): `__init__.py`, `nodes.py`, `loader.py`,
   `ops.py`, `dequant.py`. Runs *inside* ComfyUI; only depends on
   `gguf`, `sentencepiece`, `protobuf` (see root `requirements.txt`).
   It cannot be imported standalone — `nodes.py`/`ops.py` import `comfy.*`,
   which only exists inside a ComfyUI installation.
2. **Conversion toolchain** (`tools/`): standalone scripts that convert
   `.safetensors` → `.gguf` (`convert.py`), quantize via a patched
   `llama-quantize`, plus a Qt GUI (`gguf_gui.py`), a headless CLI
   (`gguf_pipeline.py`), shared orchestration (`pipeline_lib.py`), a VRAM
   analyzer (`analyze_model.py`), and fix-up/inspection helpers
   (`fix_pad.py`, `fix_5d_tensors.py`, `inspect_gguf.py`).

Docs: `README.md` (overview), `tools/README.md` (conversion setup),
`docs/CONVERSION_GUIDE.md` (per-model walkthroughs).

## Environment setup (REQUIRED before running tests or tools)

The conversion tools and the test suite need their own venv — **never**
install these deps into ComfyUI's environment. Create it with `uv`:

```bash
cd /media/john/Ubuntu_22.04_Ext/ComfyUI-GGUF
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -r tools/requirements-conversion.txt
```

- The venv lives at `<repo-root>/venv/` and is gitignored.
- If `venv/` already exists, just `source venv/bin/activate`.
- Validated combination: Python 3.11 + gguf 0.19 + safetensors 0.8 +
  torch 2.12 + numpy 2.4 + PySide6 6.11.

## Verification commands

Run from the repo root **with the venv activated**:

```bash
source venv/bin/activate
python -m unittest discover -s tools/tests -v   # full test suite (28 tests)
```

Single test module:

```bash
python -m unittest tools.tests.test_pipeline_lib -v
python -m unittest tools.tests.test_analyze_model_gguf_shape -v
```

Linting: no lint config is committed. `ruff` is available on this machine
(`~/.local/bin/ruff`); use `ruff check <files>` for advisory linting but do
not auto-fix style in untouched code — this fork tracks upstream
`city96/ComfyUI-GGUF` and gratuitous churn makes merges harder.

## Conventions & gotchas

- License headers: most files carry `# (c) City96 || Apache-2.0`. Keep them.
- Tests use stdlib `unittest` (not pytest). Test files insert `tools/` into
  `sys.path` and import modules top-level (`import pipeline_lib`), because
  `tools/` is run as loose scripts, not a package.
- `pipeline_lib.py` must stay importable **without** PySide6 / torch /
  nvidia-smi — only `gguf` at import time. GUI/GPU concerns belong in
  `gguf_gui.py`.
- `gguf_gui.py` is CRLF on disk (intentional); other `tools/*.py` are LF.
  Do not normalize line endings.
- GGUF stores tensor dims in **reversed** order vs torch. Anything reading
  shapes via `gguf.GGUFReader` must reverse them (see
  `analyze_model.read_gguf_tensors` and the regression test).
- Architecture detection lives in `tools/convert.py` (`arch_list`,
  `keys_detect`). `ModelErnie` must stay ordered before `ModelHunyuanDiT`.
  `analyze_model.py` and `loader.py` reuse this taxonomy — keep them in sync,
  including `pipeline_lib.KNOWN_ARCH_FIX_FILES`.
- The quantizer binary is expected at
  `<repo-root>/llama.cpp/build/bin/llama-quantize` (override with
  `$LLAMA_CPP_DIR`). It is a *patched* build (tag b3962 + `tools/lcpp.patch`,
  or the pre-patched `city96` branch of `Randy420Marsh/llama.cpp`).
- Conversion intermediates are large (12+ GiB F16 GGUFs). Don't run real
  conversions just to test logic — the unit tests use synthetic GGUFs.
- This is a fork. Loader bugs that also exist upstream go to
  `city96/ComfyUI-GGUF`; fork-specific features (mistral3/gemma4 TE support,
  GUI, pipeline CLI, analyzer) are maintained here.

## Devin skills & subagents

- `/run-tests` — set up/activate the venv and run the unittest suite.
- `/convert-model` — drive `tools/gguf_pipeline.py` end to end.
- `/inspect-gguf` — sanity-check a GGUF (arch, dtypes, sizes, metadata).
- `/code-review` — review changes with the `code-reviewer` subagent
  (ruff lint + manual quality review), defined in
  `.devin/agents/code-reviewer/AGENT.md`.
