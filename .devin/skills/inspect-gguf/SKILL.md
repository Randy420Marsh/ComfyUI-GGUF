---
name: inspect-gguf
description: Sanity-check a GGUF file (arch, dtype histogram, sizes, metadata) with tools/inspect_gguf.py
argument-hint: "[file.gguf]"
allowed-tools:
  - read
  - grep
  - glob
  - exec
---

Inspect the GGUF file: $1

## Environment (MANDATORY)

`inspect_gguf.py` needs the `gguf` package from the conversion venv:

```bash
cd <repo-root>
uv venv venv --python 3.11          # only if venv/ does not exist yet
source venv/bin/activate
uv pip install -r tools/requirements-conversion.txt   # only on first setup
```

## Inspection

Start with the basic report plus the size-consistency check:

```bash
python tools/inspect_gguf.py "$1" --check-sizes
```

This prints architecture, `general.file_type`, tensor count, and a
per-dtype histogram, and validates every tensor has non-zero data with
byte counts matching shape × dtype size.

Useful extra flags depending on what the user is debugging:

- `--metadata` — dump every KV metadata field (head counts, context length,
  tokenizer model, etc). Use this for text-encoder GGUFs; safetensors-based
  readers cannot open GGUFs.
- `--verbose` — list every tensor with dtype and shape.
- `--check-no-bf16` — exit non-zero if any BF16 tensors remain (guard for
  Turing / RTX 20xx targets that lack native bf16).

## Interpreting results

- `--check-sizes` failures (zero-byte tensors) are the root cause of the
  notorious `basic_ios::clear: iostream error` from llama-quantize. The
  fix is regenerating the intermediate GGUF, not patching the quantizer.
- Remember GGUF stores dims reversed vs torch convention; shapes shown by
  `--verbose` are GGUF-order.
- Report the architecture, file type, dtype histogram, and any problems
  found, with concrete next steps.
