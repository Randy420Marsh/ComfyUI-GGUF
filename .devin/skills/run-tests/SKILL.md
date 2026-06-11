---
name: run-tests
description: Set up the uv venv (if missing) and run the ComfyUI-GGUF unittest suite
argument-hint: "[test-module]"
allowed-tools:
  - read
  - grep
  - glob
  - exec
---

Run the test suite for the ComfyUI-GGUF conversion tools.

## Environment (MANDATORY)

The tests import `gguf`, `torch`, etc. from a dedicated venv at
`<repo-root>/venv/`. They will fail with `ModuleNotFoundError: No module
named 'gguf'` if you run them with the system Python.

1. If `venv/` does not exist at the repo root, create it exactly like this:

   ```bash
   cd <repo-root>
   uv venv venv --python 3.11
   source venv/bin/activate
   uv pip install -r tools/requirements-conversion.txt
   ```

2. If `venv/` already exists, just activate it:

   ```bash
   source venv/bin/activate
   ```

## Run the tests

From the repo root, with the venv activated:

```bash
python -m unittest discover -s tools/tests -v
```

If a specific test module was requested ($1), run it instead:

```bash
python -m unittest tools.tests.$1 -v
```

Available modules: `test_pipeline_lib`, `test_analyze_model_gguf_shape`.

## Reporting

- Expected baseline: 28 tests, all passing.
- Report pass/fail counts and full tracebacks for any failure.
- Note: tests use stdlib `unittest`, NOT pytest. Do not install pytest.
- Failures mentioning `comfy.*` imports mean someone tried to import the
  inference-node modules (`nodes.py`, `ops.py`) outside ComfyUI — those are
  not unit-testable standalone.
