---
name: code-review
description: Review current changes with the code-reviewer subagent (ruff lint + tests + project quality rules)
argument-hint: "[git-range]"
agent: code-reviewer
---

Review the current code changes in the ComfyUI-GGUF repository.

Scope: $ARGUMENTS
(If no argument was given, review the uncommitted working-tree changes —
staged and unstaged. If an argument was given, treat it as a git revision
range, e.g. `main...HEAD` or a single commit hash, and review that diff.)

Follow your full review procedure:

1. Collect the diff for the scope above.
2. Run `ruff check` on the changed Python files (advisory; only report
   issues on lines touched by the diff).
3. If `tools/` is touched, run the unittest suite inside the repo venv
   (`source venv/bin/activate`; create it with `uv venv venv --python 3.11`
   + `uv pip install -r tools/requirements-conversion.txt` if missing).
4. Check the project invariants (arch-list sync, pipeline_lib import
   purity, inference-side dependency limits, license headers, line
   endings, fork hygiene).

Return the structured report (Summary / Blocking issues / Ruff findings /
Test results / Suggestions) with file:line citations.
