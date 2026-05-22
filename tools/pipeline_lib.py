"""Shared GGUF-conversion pipeline used by ``gguf_gui.py`` and the
``gguf_pipeline.py`` CLI.

The pipeline is:

    Step 0: Pre-process >4D tensors so handle_nd_tensor never fires.
    Step 1: Convert .safetensors -> _f16.gguf via convert.py.
    Step 2: fix_pad.py (only when 1-D x_pad_token / cap_pad_token detected).
    Step 3: llama-quantize -> _<quant>.gguf.
    Step 4: Optional re-attach of 5-D tensors via fix_5d_tensors.py for
            hyvid / wan archs that bypassed Step 0 (legacy workflows).
    Step 5: Cleanup of intermediate files.

The single entry point is :func:`run_pipeline`, which accepts a ``log``
callable so both the Qt GUI (``log_signal.emit``) and the CLI (``print``)
can plug their own output mechanism in.

Why this module exists
----------------------
``gguf_gui.py`` originally embedded all of this in ``ConversionThread.run``.
Adding a CLI one-shot would mean duplicating that logic; refactoring to
this shared module instead keeps a single source of truth for the
pre-flight check, the Step 2 skip heuristic, and the 5-D handling.

Line endings: LF, like other tools/* importable Python modules
(``analyze_model.py``, ``fix_pad.py``).  ``gguf_gui.py`` itself stays CRLF
to preserve its existing on-disk encoding.
"""

import os
import subprocess
import sys
from typing import Callable, Optional

# Importable without PySide6 / torch / nvidia-smi — keep this module pure
# and let callers handle UI / GPU-detection concerns.
from gguf import GGUFReader


# ── Path resolution ────────────────────────────────────────────────────────

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)

CONVERT_PY = os.path.join(_TOOLS_DIR, "convert.py")
FIX_PAD_PY = os.path.join(_TOOLS_DIR, "fix_pad.py")
FIX_5D_PY = os.path.join(_TOOLS_DIR, "fix_5d_tensors.py")


def _resolve_llama_cpp_dir() -> str:
    """Locate the llama.cpp clone that contains the patched llama-quantize.

    Search order:
      1. ``$LLAMA_CPP_DIR`` (explicit override)
      2. ``<repo_root>/llama.cpp`` (the layout tools/README.md documents)
      3. ``./llama.cpp`` relative to the current working directory (legacy
         behaviour, kept so users who launched the GUI from the repo
         root in older versions still work)

    Returns an absolute path.  The directory is not required to exist at
    import time — existence is checked later when ``llama-quantize`` is
    actually invoked, so this module can still load when the build is
    missing.
    """
    override = os.environ.get("LLAMA_CPP_DIR")
    if override:
        return os.path.abspath(override)
    repo_local = os.path.join(_REPO_ROOT, "llama.cpp")
    if os.path.isdir(repo_local):
        return repo_local
    return os.path.abspath(os.path.join(os.getcwd(), "llama.cpp"))


LLAMA_CPP_DIR = _resolve_llama_cpp_dir()
LLAMA_QUANTIZE_BIN = os.path.join(
    LLAMA_CPP_DIR, "build", "bin", "llama-quantize"
)
LLAMA_QUANTIZE_BIN_WIN = os.path.join(
    LLAMA_CPP_DIR, "build", "bin", "Release", "llama-quantize.exe"
)
LD_PATH_BUILD_SRC = os.path.join(LLAMA_CPP_DIR, "build", "src")
LD_PATH_BUILD_GGML_SRC = os.path.join(LLAMA_CPP_DIR, "build", "ggml", "src")


def locate_llama_quantize() -> tuple[str, Optional[str]]:
    """Return ``(bin_path, error_or_None)`` for the patched llama-quantize.

    Tries the Windows Release path first on Windows, then the POSIX path
    on every platform.  Returns the *expected* path even on miss so the
    error message can quote it back to the user.
    """
    if os.name == "nt" and os.path.exists(LLAMA_QUANTIZE_BIN_WIN):
        return LLAMA_QUANTIZE_BIN_WIN, None
    if os.path.exists(LLAMA_QUANTIZE_BIN):
        return LLAMA_QUANTIZE_BIN, None
    expected = LLAMA_QUANTIZE_BIN_WIN if os.name == "nt" else LLAMA_QUANTIZE_BIN
    override_msg = (
        f"$LLAMA_CPP_DIR override active -> {LLAMA_CPP_DIR!r}\n"
        if os.environ.get("LLAMA_CPP_DIR")
        else (
            "Expected layout: <ComfyUI-GGUF repo root>/llama.cpp/build/bin/llama-quantize\n"
            "(set $LLAMA_CPP_DIR to override if llama.cpp lives elsewhere).\n"
        )
    )
    err = (
        f"llama-quantize binary not found at {expected}.\n"
        f"{override_msg}"
        "Build it first: see tools/README.md \u00a7 3 or the wiki page\n"
        "Build-llama-quantize. The shortcut is:\n"
        "  git clone -b city96 https://github.com/Randy420Marsh/llama.cpp.git\n"
        "  cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release \\\n"
        "    -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON\n"
        "  cmake --build build --config Release -j --target llama-quantize\n"
    )
    return expected, err


# ── Quantization output types (verbatim from llama-quantize @ b3962) ──────

LLAMA_QUANTIZE_TYPES = [
    ("Q4_0",     "4.34G, +0.4685 ppl @ Llama-3-8B"),
    ("Q4_1",     "4.78G, +0.4511 ppl @ Llama-3-8B"),
    ("Q5_0",     "5.21G, +0.1316 ppl @ Llama-3-8B"),
    ("Q5_1",     "5.65G, +0.1062 ppl @ Llama-3-8B"),
    ("IQ2_XXS",  "2.06 bpw quantization"),
    ("IQ2_XS",   "2.31 bpw quantization"),
    ("IQ2_S",    "2.5  bpw quantization"),
    ("IQ2_M",    "2.7  bpw quantization"),
    ("IQ1_S",    "1.56 bpw quantization"),
    ("IQ1_M",    "1.75 bpw quantization"),
    ("TQ1_0",    "1.69 bpw ternarization"),
    ("TQ2_0",    "2.06 bpw ternarization"),
    ("Q2_K",     "2.96G, +3.5199 ppl @ Llama-3-8B"),
    ("Q2_K_S",   "2.96G, +3.1836 ppl @ Llama-3-8B"),
    ("IQ3_XXS",  "3.06 bpw quantization"),
    ("IQ3_S",    "3.44 bpw quantization"),
    ("IQ3_M",    "3.66 bpw quantization mix"),
    ("Q3_K",     "alias for Q3_K_M"),
    ("IQ3_XS",   "3.3 bpw quantization"),
    ("Q3_K_S",   "3.41G, +1.6321 ppl @ Llama-3-8B"),
    ("Q3_K_M",   "3.74G, +0.6569 ppl @ Llama-3-8B"),
    ("Q3_K_L",   "4.03G, +0.5562 ppl @ Llama-3-8B"),
    ("IQ4_NL",   "4.50 bpw non-linear quantization"),
    ("IQ4_XS",   "4.25 bpw non-linear quantization"),
    ("Q4_K",     "alias for Q4_K_M"),
    ("Q4_K_S",   "4.37G, +0.2689 ppl @ Llama-3-8B"),
    ("Q4_K_M",   "4.58G, +0.1754 ppl @ Llama-3-8B  (recommended default for 8 GB VRAM)"),
    ("Q5_K",     "alias for Q5_K_M"),
    ("Q5_K_S",   "5.21G, +0.1049 ppl @ Llama-3-8B"),
    ("Q5_K_M",   "5.33G, +0.0569 ppl @ Llama-3-8B  (sweet spot if it fits)"),
    ("Q6_K",     "6.14G, +0.0217 ppl @ Llama-3-8B"),
    ("Q8_0",     "7.96G, +0.0026 ppl @ Llama-3-8B  (~8 bpw; rarely fits on 8 GB)"),
    ("Q4_0_4_4", "4.34G, ARM-only repack of Q4_0 (do not use on x86 GPUs)"),
    ("Q4_0_4_8", "4.34G, ARM-only repack of Q4_0 (do not use on x86 GPUs)"),
    ("Q4_0_8_8", "4.34G, ARM-only repack of Q4_0 (do not use on x86 GPUs)"),
    ("F16",      "~14 G; no quantization, half-precision storage"),
    ("BF16",     "~14 G; no quantization, bf16 storage (Ampere+ only at runtime)"),
    ("F32",      "~28 G; no quantization, full float32 (debugging only)"),
    ("COPY",     "copy tensors verbatim, no quantization (debugging)"),
]
QUANTIZE_TYPE_NAMES = [name for name, _desc in LLAMA_QUANTIZE_TYPES]
DEFAULT_QUANT_TYPE = "Q4_K_M"


# Every arch that may produce a fix_5d_tensors_*.safetensors via convert.py
# (kept in sync with ``tools/convert.py`` ``arch_list``).
KNOWN_ARCH_FIX_FILES = [
    "hyvid", "wan", "hunyuan", "flux", "sd3", "aura",
    "ltxv", "hidream", "cosmos", "lumina2", "ernie",
]


# Archs for which Step 0 may legitimately leave a fix_5d_tensors_*.safetensors
# behind even with the pre-fold in place (because the user might be running
# the CLI in a mode that bypasses Step 0).  Currently HyVid + Wan.
ARCHS_NEEDING_5D_REATTACH = ("hyvid", "wan")


# ── Shell helpers ──────────────────────────────────────────────────────────

def shell_quote(path: str) -> str:
    """Quote a path for safe interpolation into a shell command string."""
    if os.name == "nt":
        return '"' + path.replace('"', '\\"') + '"'
    return "'" + path.replace("'", "'\\''") + "'"


def build_subprocess_env() -> dict:
    """Return an environment dict with ``LD_LIBRARY_PATH`` /
    ``DYLD_LIBRARY_PATH`` pointing at the llama.cpp build tree.

    The patched ``llama-quantize`` links against shared ``libggml.so`` /
    ``libllama.so`` inside ``build/`` so it can't run from anywhere
    without this.  Set both vars so the same code path works on macOS
    too.
    """
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        LD_PATH_BUILD_SRC + os.pathsep
        + LD_PATH_BUILD_GGML_SRC + os.pathsep
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["DYLD_LIBRARY_PATH"] = (
        LD_PATH_BUILD_SRC + os.pathsep
        + LD_PATH_BUILD_GGML_SRC + os.pathsep
        + env.get("DYLD_LIBRARY_PATH", "")
    )
    return env


def stream_command(cmd: str, env: dict, log: Callable[[str], None]) -> int:
    """Run a shell command and stream stdout/stderr line-by-line to ``log``."""
    process = subprocess.Popen(
        cmd, shell=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    buffer = ""
    while True:
        char = process.stdout.read(1)
        if not char and process.poll() is not None:
            break
        if char in ("\r", "\n"):
            cleaned = buffer.strip()
            if cleaned:
                log(cleaned)
            buffer = ""
        else:
            buffer += char
    if buffer.strip():
        log(buffer.strip())
    return process.returncode


# ── GGUF metadata helpers ──────────────────────────────────────────────────

def needs_pad_fix(f16_path: str, log: Callable[[str], None] = lambda _m: None) -> bool:
    """Return True iff the F16 GGUF has a 1-D ``x_pad_token`` /
    ``cap_pad_token``.  For all other models ``fix_pad.py`` is a no-op,
    and skipping it avoids an expensive (and occasionally lossy) round-trip
    through GGUFReader/Writer (the cause of the Z-Image 0.36
    ``basic_ios::clear: iostream error`` we fixed in PR #16).
    """
    try:
        reader = GGUFReader(f16_path, "r")
        for t in reader.tensors:
            if t.name in ("x_pad_token", "cap_pad_token") and len(t.shape) == 1:
                return True
        return False
    except Exception as exc:
        log(f"  (could not inspect {f16_path} for pad tokens: {exc}; "
            f"running fix_pad as a precaution)")
        return True


def get_gguf_arch(path: str) -> Optional[str]:
    """Read ``general.architecture`` from a GGUF file, returning the
    arch string (e.g. ``'flux'``, ``'hyvid'``) or ``None`` on error."""
    try:
        reader = GGUFReader(path, "r")
        field = reader.get_field("general.architecture")
        if field is None:
            return None
        return str(field.parts[field.data[-1]], encoding="utf-8")
    except Exception:
        return None


# ── Step 0: Pre-process >4D tensors ────────────────────────────────────────

def preprocess_5d_tensors(src: str, temp_dir: str,
                          log: Callable[[str], None]) -> str:
    """Fold any (a, b, c, d, e) tensor to (a*b, c, d, e) before convert.py
    sees the file.  Returns the path to pass to convert.py (the original
    when no folding was needed, or a ``_5dfixed.safetensors`` copy in
    ``temp_dir`` otherwise).

    Why this is needed
    ------------------
    ``convert.py``'s ``handle_nd_tensor`` saves the offending tensor to
    ``fix_5d_tensors_{arch}.safetensors`` and skips it, so the F16 GGUF
    ends up missing that tensor entirely (HyVid / Wan).  For every other
    architecture the base class raises ``NotImplementedError`` and
    conversion aborts.  Either way, a 5-D tensor reaching
    ``llama-quantize`` triggers
    ``GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS)`` and a core
    dump.
    """
    try:
        import torch  # noqa: F401 — ensure safetensors dtype support
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        log(f"  [!] torch/safetensors not importable: {exc}")
        log("  [!] Skipping 5D scan. If quantization crashes with")
        log("      GGML_ASSERT(n_dims) check your Python environment.")
        return src

    log("  Scanning tensor ranks in source file\u2026")
    tensors = load_file(src)

    scaled_fp8_weights = [
        k for k in tensors.keys() if k.endswith(".weight_scale")
    ]
    if scaled_fp8_weights:
        log(
            f"  Detected ComfyUI scaled-fp8 quantization on "
            f"{len(scaled_fp8_weights)} weight(s); convert.py will dequantize them."
        )

    over_4d = [k for k, v in tensors.items() if len(v.shape) > 4]
    if not over_4d:
        log("  \u2713 All tensors \u2264 4D \u2014 no pre-processing needed.")
        return src

    log(f"  Found {len(over_4d)} tensor(s) with >4 dimensions. Folding\u2026")
    fixed = {}
    for key, tensor in tensors.items():
        if len(tensor.shape) <= 4:
            fixed[key] = tensor
            continue
        t = tensor
        while len(t.shape) > 4:
            new_shape = (t.shape[0] * t.shape[1],) + t.shape[2:]
            t = t.reshape(new_shape)
        log(f"    Folded  {key}  {list(tensor.shape)}  \u2192  {list(t.shape)}")
        fixed[key] = t

    base = os.path.basename(src).replace(".safetensors", "_5dfixed.safetensors")
    fixed_src = os.path.join(temp_dir, base)
    save_file(fixed, fixed_src)
    log(f"  Fixed source written: {fixed_src}")
    return fixed_src


# ── Step 4 (optional): re-attach 5-D tensors via fix_5d_tensors.py ────────

def maybe_reattach_5d(quantized_path: str, log: Callable[[str], None],
                      env: dict) -> Optional[str]:
    """If ``./fix_5d_tensors_<arch>.safetensors`` exists for the quantized
    file's arch, run ``fix_5d_tensors.py`` to merge those tensors back
    into the quantized output.  Returns the new output path (a sibling
    of ``quantized_path``) on success, or ``None`` when nothing needed
    re-attaching.

    Note: with Step 0's pre-fold this path is normally a no-op for runs
    started from a clean state.  It exists for the case where the
    cached ``_f16.gguf`` was produced by an older ``convert.py`` run
    that bypassed Step 0 (so ``handle_nd_tensor`` fired and left a fix
    file behind), or where the user manually invoked ``convert.py``.
    """
    arch = get_gguf_arch(quantized_path)
    if arch not in ARCHS_NEEDING_5D_REATTACH:
        return None
    fix_path = f"./fix_5d_tensors_{arch}.safetensors"
    if not os.path.exists(fix_path):
        return None

    base, ext = os.path.splitext(quantized_path)
    reattached = f"{base}_5dfix{ext}"
    log(f"  Detected leftover {fix_path}; re-attaching 5-D tensors\u2026")
    cmd = (
        f"{shell_quote(sys.executable)} {shell_quote(FIX_5D_PY)} "
        f"--src {shell_quote(quantized_path)} "
        f"--dst {shell_quote(reattached)} "
        f"--fix {shell_quote(fix_path)} --overwrite"
    )
    rc = stream_command(cmd, env, log)
    if rc != 0:
        log("  [!] fix_5d_tensors.py failed; leaving quantized output untouched.")
        return None
    os.replace(reattached, quantized_path)
    try:
        os.remove(fix_path)
    except OSError:
        pass
    log(f"  \u2713 Re-attached 5-D tensors into {quantized_path}.")
    return quantized_path


# ── Main entry point ──────────────────────────────────────────────────────

def run_pipeline(
    src: str,
    temp_dir: str,
    out_dir: str,
    quant_type: str = DEFAULT_QUANT_TYPE,
    dtype_cli: str = "auto",
    dtype_reason: str = "",
    log: Callable[[str], None] = print,
    cleanup: bool = True,
    auto_5d_reattach: bool = True,
) -> str:
    """Run the full conversion pipeline.

    Returns the absolute path of the final quantized ``.gguf``.  Raises
    ``RuntimeError`` (or ``FileNotFoundError`` / etc.) on failure; the
    caller is responsible for translating that into a GUI alert / a
    non-zero exit code.

    Parameters
    ----------
    src : path to the source ``.safetensors``.
    temp_dir : directory for intermediate ``_f16.gguf`` and ``_5dfixed`` files.
    out_dir  : directory where the final quantized ``.gguf`` lands.
    quant_type : one of :data:`QUANTIZE_TYPE_NAMES`.
    dtype_cli  : ``'auto'`` / ``'fp16'`` / ``'bf16'``; maps to
                 ``convert.py --dtype``.  ``'auto'`` is a no-op (no flag).
    dtype_reason : free-form explanation that will be logged before
                   conversion (the GUI uses this to print the GPU-capability
                   detection rationale; the CLI typically leaves it empty).
    log        : callable for streaming progress output.  Defaults to
                 :func:`print`; the GUI passes its Qt signal.emit.
    cleanup    : delete intermediate files when True (default).
    auto_5d_reattach : run ``fix_5d_tensors.py`` after quantization if a
                       leftover ``fix_5d_tensors_<arch>.safetensors`` is
                       present (default True).
    """
    if quant_type not in QUANTIZE_TYPE_NAMES:
        raise ValueError(
            f"Unknown quant type {quant_type!r}; "
            f"must be one of {QUANTIZE_TYPE_NAMES}"
        )
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Source not found: {src}")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Pre-flight: confirm the patched llama-quantize exists.
    _, quantize_err = locate_llama_quantize()
    if quantize_err:
        raise RuntimeError(quantize_err)

    env = build_subprocess_env()
    filename = os.path.basename(src)
    f16_tmp = os.path.join(temp_dir, filename + "_f16.gguf")
    fixed_tmp = os.path.join(temp_dir, filename + "_f16_fixed.gguf")
    final_out = os.path.join(
        out_dir,
        filename.replace(".safetensors", f"_{quant_type}.gguf"),
    )

    # ── Step 0 ─────────────────────────────────────────────────────────
    log("=== Step 0: High-dimensional tensor check ===")
    src_for_convert = preprocess_5d_tensors(src, temp_dir, log)

    # Clean up stale convert.py fix files from previous runs.  ModelHyVid
    # raises RuntimeError if the fix file already exists, which would
    # abort Step 1 on a re-run.
    for arch in KNOWN_ARCH_FIX_FILES:
        stale = f"./fix_5d_tensors_{arch}.safetensors"
        if os.path.exists(stale):
            os.remove(stale)
            log(f"  Removed stale fix file: {stale}")

    # ── Step 1 ─────────────────────────────────────────────────────────
    skip_convert = False
    if os.path.exists(f16_tmp):
        log("\n=== Step 1: Verifying cached F16 file ===")
        log(f"  Found: {f16_tmp}")
        try:
            reader = GGUFReader(f16_tmp, "r")
            if len(reader.tensors) > 0:
                log(f"  \u2713 Valid ({len(reader.tensors)} tensors). Skipping conversion.")
                skip_convert = True
            else:
                log("  \u2717 Empty tensor list \u2014 treating as corrupt. Reconverting.")
        except Exception as exc:
            log(f"  \u2717 Cannot read file ({exc}). Reconverting.")

    if not skip_convert:
        log("\n=== Step 1: Converting to F16 GGUF ===")
        if dtype_reason:
            log(f"  {dtype_reason}")
        dtype_arg = "" if dtype_cli == "auto" else f" --dtype {dtype_cli}"
        cmd = (
            f"{shell_quote(sys.executable)} {shell_quote(CONVERT_PY)} "
            f"--src {shell_quote(src_for_convert)} "
            f"--dst {shell_quote(f16_tmp)}{dtype_arg}"
        )
        if stream_command(cmd, env, log) != 0:
            raise RuntimeError("F16 conversion failed. See log above.")

    # ── Step 2 ─────────────────────────────────────────────────────────
    log("\n=== Step 2: Applying padding shape fix ===")
    if needs_pad_fix(f16_tmp, log):
        cmd = (
            f"{shell_quote(sys.executable)} {shell_quote(FIX_PAD_PY)} "
            f"{shell_quote(f16_tmp)}"
        )
        if stream_command(cmd, env, log) != 0:
            raise RuntimeError("fix_pad.py failed. See log above.")
        quantize_input = fixed_tmp
    else:
        log("  No 1-D pad tokens detected -- skipping fix_pad.py. "
            "Quantizing the F16 file directly.")
        quantize_input = f16_tmp

    # ── Step 3 ─────────────────────────────────────────────────────────
    log(f"\n=== Step 3: Quantizing to {quant_type} ===")
    quantize_bin, quantize_err = locate_llama_quantize()
    if quantize_err:
        raise RuntimeError(quantize_err)
    cmd = (
        f"{shell_quote(quantize_bin)} "
        f"{shell_quote(quantize_input)} {shell_quote(final_out)} "
        f"{quant_type}"
    )
    if stream_command(cmd, env, log) != 0:
        raise RuntimeError(
            "llama-quantize failed.\n\n"
            "If the log shows  GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS):\n"
            "  A tensor with >4 dimensions reached the quantizer via the cached F16\n"
            "  file (built before this fix was in place).\n\n"
            "  Solution: delete the stale .gguf files from your temp folder\n"
            "  and run again. Step 0 will fold those tensors before Step 1 runs."
        )

    # ── Step 4: optional 5-D re-attach for HyVid / Wan legacy flows ───
    if auto_5d_reattach:
        log("\n=== Step 4: Checking for 5-D tensor re-attach ===")
        if maybe_reattach_5d(final_out, log, env) is None:
            log("  Nothing to re-attach (Step 0 pre-fold or non-HyVid/Wan arch).")

    # ── Step 5: cleanup ───────────────────────────────────────────────
    if cleanup:
        log("\n=== Step 5: Cleaning up temp files ===")
        for tmp in (f16_tmp, fixed_tmp):
            if os.path.exists(tmp):
                os.remove(tmp)
                log(f"  Removed: {tmp}")
        fixed_src_path = os.path.join(
            temp_dir,
            filename.replace(".safetensors", "_5dfixed.safetensors"),
        )
        if os.path.exists(fixed_src_path):
            os.remove(fixed_src_path)
            log(f"  Removed: {fixed_src_path}")

    # Final verification — open the output once so we fail loud if it
    # got truncated mid-write.
    try:
        reader = GGUFReader(final_out)
        log(f"\n\u2713 Output verified: {len(reader.tensors)} tensors written successfully.")
    except Exception as exc:
        log(f"\n\u26a0 Output verification warning: {exc}")

    return final_out
