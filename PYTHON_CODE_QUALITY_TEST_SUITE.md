# PYTHON_CODE_QUALITY_TEST_SUITE.md

Reusable playbook for setting up a Python code-quality and test pipeline in
**any** Python project. Written as step-by-step instructions for an AI coding
agent (Devin, etc.), but equally usable by a human. Copy this file into a new
repo and follow it top to bottom — adapt the placeholder paths
(`<src-dir>`, `<tests-dir>`, `<requirements-file>`) to the project.

Distilled from the quality work done on ComfyUI-GGUF (venv isolation, ruff
tiered linting, stdlib unittest regression suite, warnings-as-errors,
subprocess/resource hygiene, fork-aware review rules).

---

## 0. Ground rules for the agent

- **Never install project dependencies into the system Python or another
  app's environment.** Always use a dedicated venv at `<repo-root>/venv/`.
- **Do not auto-fix style in code you did not touch.** If the project is a
  fork or tracks an upstream, gratuitous churn makes merges harder. Lint
  findings on untouched lines are *advisory only*.
- **Preserve existing conventions**: license headers, line endings
  (some files may be intentionally CRLF), naming style, test framework
  choice. Detect before you change.
- **Verify before claiming done**: every change set must pass the full
  pipeline in §5 before being committed.

---

## 1. Environment setup (once per machine / clone)

Use [`uv`](https://docs.astral.sh/uv/) — it is faster than `python -m venv`
+ `pip` and resolves interpreters automatically. Fall back to plain
`venv`/`pip` only if `uv` is unavailable and can't be installed.

```bash
cd <repo-root>
uv venv venv --python 3.11          # pin a known-good interpreter
source venv/bin/activate            # Windows: venv\Scripts\activate.bat
uv pip install -r <requirements-file>
```

Notes:
- Pin the Python minor version that the project documents (or the newest
  stable if undocumented) and record it in the project's `AGENTS.md`.
- If there are separate runtime vs. dev/tooling requirements, install both
  (e.g. `requirements.txt` + `requirements-dev.txt`).
- `venv/` must be gitignored. Check `.gitignore`; add it if missing.
- Record the validated version combination (python + key deps) in
  `AGENTS.md` once the suite passes, so future runs can reproduce it.

---

## 2. Static checks (no test execution needed)

Run these first — they are fast and catch a large class of bugs.

### 2.1 Syntax / bytecode compile check

Every Python file must at minimum compile. This catches syntax errors in
files that no test imports:

```bash
find <src-dir> -name '*.py' -not -path '*/venv/*' -print0 \
  | xargs -0 python -m py_compile
```

### 2.2 Ruff — tiered linting

Install once (outside the venv is fine): `uv tool install ruff` or
`pipx install ruff`. If the repo has a committed ruff/flake8 config, obey
it and skip the tiering below.

Run in three tiers, strictest treatment for the narrowest scope:

| Tier | Command | Policy |
|------|---------|--------|
| 1. Default rules (E/F: pyflakes + pycodestyle errors) | `ruff check <changed-files>` | **Must be clean** on files you created or modified. |
| 2. Bug patterns | `ruff check --select B,PLE,RUF <changed-files>` | Must be clean on your changes; report (don't fix) pre-existing hits elsewhere. |
| 3. Whole-repo advisory | `ruff check .` | Report findings; only fix with explicit approval. |

Common findings worth fixing on sight (in code you own/touch):

- `F401` unused imports — remove, or `# noqa: F401` with a comment when the
  import is an intentional availability probe.
- `E721` `type(x) == type(y)` — use `isinstance()` or `is`.
- `E722` bare `except:` — catch specific exceptions, at minimum `Exception`.
- `E731` lambda assignment — use `def`.
- `B006` **mutable default arguments** (`def f(x={})`) and class-level
  mutable defaults — replace with `None` + initialize inside.
- `B008` function call in default argument.

### 2.3 Optional: type checking

If the project already has type hints or a `mypy`/`pyright` config, run it.
Do **not** introduce a type checker into an untyped codebase without being
asked.

---

## 3. Test suite layout (creating one from scratch)

Prefer **stdlib `unittest`** for tooling/script repos with no existing test
infra — zero extra dependencies and `python -m unittest discover` just
works. Use `pytest` only if the project already uses it.

Layout:

```
<repo-root>/
├── <src-dir>/                # e.g. tools/, src/, mypackage/
│   ├── module_a.py
│   └── tests/
│       ├── __init__.py       # empty; makes discovery reliable
│       ├── test_module_a.py
│       └── test_regressions.py
└── venv/                     # gitignored
```

If `<src-dir>` is loose scripts (not an installed package), make each test
file self-locating so tests run from the repo root without packaging:

```python
import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # <src-dir>
import module_a  # noqa: E402
```

### 3.1 What to test (priority order)

1. **Pure logic helpers** — naming/parsing/path manipulation, math,
   format detection. Cheap, high value, no I/O mocking needed.
2. **Regression tests for every bug you fix.** A bug fix without a test
   that fails-before / passes-after is incomplete. Name them descriptively:
   `test_double_extension_only_touches_last`, not `test_fix2`.
3. **Subprocess wrappers** — exit-code propagation, output streaming,
   `\r` progress-line handling, EOF when the child closes stdout early,
   and that pipes/handles are closed (see §4).
4. **CLI argument parsing** — defaults, choice validation, that `--help`
   stays in sync with the canonical constant lists.
5. **Constant/taxonomy consistency** — when one list is the "source of
   truth" mirrored elsewhere (arch lists, registries, enum mirrors), add a
   test asserting the mirrors are subsets/equal. This converts silent
   drift into a test failure.
6. **Error paths** — missing files, unreadable input, helpful error
   messages (assert the message contains the actionable hint).

### 3.2 Test data

- Use synthetic fixtures generated in `setUp`/`setUpClass` (e.g. tiny
  generated files), never multi-GB real assets. Keep fixture creation in
  the test file itself or a small `tests/fixtures.py`.
- Use `tempfile.TemporaryDirectory()` and clean up via context managers or
  `addCleanup` — tests must not litter the repo.

---

## 4. Resource & robustness hygiene checks

These caught real bugs in practice; run them every time:

### 4.1 Warnings escalated to errors

```bash
python -W error::ResourceWarning -m unittest discover -s <tests-dir> -v
```

`ResourceWarning` failures mean leaked file handles / subprocess pipes
(e.g. a `Popen` stdout never closed). Fix the leak, don't suppress the
warning. Optionally also run with `-W error::DeprecationWarning` and
triage the results.

### 4.2 Subprocess rules (review checklist)

- Always pass `args` as a list, or shell-quote every interpolated path
  (`shlex.quote`) if `shell=True` is unavoidable.
- Read the child's output to EOF *and* `wait()` for it; close the stdout
  pipe explicitly when streaming manually.
- Propagate the child's exit code — never swallow a non-zero return.
- Never modify user input files in place; write outputs to new paths.

---

## 5. The full verification pipeline (run before every commit)

```bash
source venv/bin/activate

# 1. Syntax
find <src-dir> -name '*.py' -not -path '*/venv/*' -print0 | xargs -0 python -m py_compile

# 2. Lint (changed files must be clean)
ruff check $(git diff --name-only --diff-filter=ACM HEAD -- '*.py')

# 3. Tests
python -m unittest discover -s <tests-dir> -v

# 4. Tests with resource warnings escalated
python -W error::ResourceWarning -m unittest discover -s <tests-dir>
```

Record the expected baseline (e.g. "37 tests, all passing") in `AGENTS.md`
and update it whenever tests are added, so a future agent can detect
unexpected skips/losses immediately.

---

## 6. Code-review checklist (manual, on the diff)

Review `git diff` against these categories. For an AI agent: do this as a
separate read-only pass (ideally a dedicated read-only subagent that cannot
edit files) and produce a structured report — Summary / Blocking issues
(with `file:line`) / Lint findings / Test results / Suggestions.

**Correctness**
- Logic errors, off-by-one, inverted conditions, wrong dimension/byte
  order, `==` vs `=` in chained comparisons.
- Edge cases: empty input, single element, unicode paths, paths with
  spaces, files without extensions, double extensions.

**API & state**
- Mutable class-level defaults shared across instances.
- Functions that mutate their arguments without documenting it.
- Single-source-of-truth constants mirrored elsewhere without a sync test.

**Robustness**
- Bare excepts hiding real failures; error messages that don't tell the
  user what to do.
- Resource handles (files, sockets, subprocess pipes) without
  `with`/explicit close.

**Security**
- No secrets/keys in code or logs. No shell injection via unquoted paths.
  No destructive operations on user data.

**Project hygiene**
- License headers preserved. Line endings unchanged. Comments not
  added/removed gratuitously. Docs updated for user-facing changes.
- Fork-aware: minimal diff against upstream-owned code.

---

## 7. Bootstrapping checklist for a NEW project (agent quickstart)

Execute in order; each step is idempotent:

1. `uv venv venv --python 3.11 && source venv/bin/activate`
2. `uv pip install -r <requirements-file>` (create the file if missing,
   listing only direct deps with minimum versions, e.g. `requests>=2.31`).
3. Ensure `.gitignore` covers `venv/`, `__pycache__/`, `*.pyc`.
4. Create `<tests-dir>/__init__.py` and a first smoke test that imports
   every module under `<src-dir>` (catches import-time crashes):

   ```python
   import importlib, pkgutil, unittest

   class ImportSmokeTest(unittest.TestCase):
       def test_all_modules_import(self):
           for m in ["module_a", "module_b"]:   # list project modules
               importlib.import_module(m)
   ```

5. Run the §5 pipeline; fix until green.
6. Write `AGENTS.md` at the repo root recording: venv setup commands, the
   exact test command, the passing baseline count, lint policy, and any
   project-specific invariants/gotchas discovered.
7. (Optional) Add agent skills mirroring this doc: a `run-tests` skill
   (venv setup + unittest discovery + baseline), and a read-only
   `code-reviewer` subagent enforcing §6.
8. Add regression tests with every subsequent bug fix — the suite should
   only ever grow.

---

## 8. Anti-patterns to refuse

- Installing pytest (or any new framework) into a project that already
  uses unittest, "just to run tests".
- Blanket `ruff --fix` / autoformatting across a repo that has no
  committed formatter config.
- Marking work complete with failing or skipped tests.
- Suppressing warnings instead of fixing the underlying leak.
- Tests that depend on huge local assets, the network, or wall-clock
  timing tighter than ~1s.
- Deleting or rewriting upstream-owned code for style reasons in a fork.
