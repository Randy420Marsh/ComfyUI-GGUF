import sys
import json
import os
import subprocess
import traceback
from urllib.parse import unquote
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QLineEdit, QPushButton, QLabel, QFileDialog,
                               QPlainTextEdit, QHBoxLayout, QProgressBar,
                               QComboBox, QDialog, QDialogButtonBox,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QTextEdit, QMessageBox)
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QColor, QFont
from gguf import GGUFReader

# analyze_model lives in the same tools/ directory; import it directly
# rather than via tools.* so the GUI script remains runnable as a
# top-level `python tools/gguf_gui.py` from the repo root.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _TOOLS_DIR)
import analyze_model

CONFIG_FILE = "settings.json"

# Paths to the conversion helper scripts and the llama.cpp build output are
# all derived from this file's own location so the GUI works regardless of
# the user's working directory (repo root, tools/, or anywhere else).
#
# llama.cpp is expected to be cloned and built inside the repo root, i.e.
# <repo_root>/llama.cpp/build/... -- this matches the layout in
# tools/README.md. Set LLAMA_CPP_DIR to override.
CONVERT_PY = os.path.join(_TOOLS_DIR, "convert.py")
FIX_PAD_PY = os.path.join(_TOOLS_DIR, "fix_pad.py")


def _resolve_llama_cpp_dir():
    """Locate the llama.cpp clone that contains the patched llama-quantize.

    Search order:
      1. $LLAMA_CPP_DIR (explicit override)
      2. <repo_root>/llama.cpp   (the layout tools/README.md documents)
      3. ./llama.cpp relative to the current working directory (legacy
         behaviour, kept so users who launched the GUI from the repo
         root in older versions still work)

    Returns an absolute path. The directory is not required to exist at
    import time -- existence is checked later when llama-quantize is
    actually invoked, so the GUI can still load when the build is missing.
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


def _shell_quote(path):
    """Quote a path for safe interpolation into a shell command string.

    The GUI uses subprocess.Popen(..., shell=True) for streaming, so any
    path that contains spaces, parentheses, or other shell-significant
    characters needs explicit quoting.
    """
    if os.name == "nt":
        return '"' + path.replace('"', '\\"') + '"'
    return "'" + path.replace("'", "'\\''") + "'"

# Output GGUF types supported by llama-quantize at the tag `tools/lcpp.patch`
# targets (b3962). Each entry is (name, description) — the description is
# verbatim from quantize.cpp:quant_options at that tag. Order chosen to
# match the upstream CLI help so the GUI mirrors what `llama-quantize`
# itself shows.
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
DEFAULT_QUANT_TYPE = "Q4_K_M"

# Minimum NVIDIA compute capability for native BF16 tensor-core support.
# Ampere = 8.0 (A100), 8.6 (RTX 30xx), 8.9 (Ada/RTX 40xx), 9.0 (Hopper/H100),
# 10.x (Blackwell/RTX 50xx, B100). Turing (CC 7.5, e.g. RTX 20xx) and earlier
# have no native BF16 — BF16 tensors are upcast to fp32 at inference.
BF16_MIN_COMPUTE_CAP = 8.0


def detect_nvidia_gpu():
    """Probe nvidia-smi for GPU name + compute capability.

    Returns (gpu_name, compute_capability_str) for the *lowest*-CC GPU on
    the system (because the model has to be runnable on whichever GPU the
    user picks). Returns (None, None) if nvidia-smi is unavailable, fails,
    or reports no GPUs (CPU-only / ROCm / Apple Silicon).
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, None
    if proc.returncode != 0:
        return None, None

    lowest_cc = None
    lowest_name = None
    lowest_cc_str = None
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        name, cc_str = parts[0], parts[1]
        try:
            cc = float(cc_str)
        except ValueError:
            continue
        if lowest_cc is None or cc < lowest_cc:
            lowest_cc = cc
            lowest_name = name
            # Preserve the original 'X.Y' string from nvidia-smi rather than
            # reformatting; '9.0' is more meaningful than '9' to the user.
            lowest_cc_str = cc_str

    if lowest_name is None:
        return None, None
    return lowest_name, lowest_cc_str


def detect_nvidia_vram_gb():
    """Return the smallest VRAM (in GB) across all visible NVIDIA GPUs.

    Used by the Analyze button to decide which quant fits. Returns None
    when nvidia-smi is unavailable -- analysis still runs in that case,
    it just won't fill in the 'fits' / recommendation columns.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None

    smallest = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            # nvidia-smi reports memory.total in MiB.
            mib = float(line)
        except ValueError:
            continue
        gb = mib / 1024.0
        if smallest is None or gb < smallest:
            smallest = gb
    return smallest


def resolve_auto_dtype():
    """Decide the --dtype value to pass to convert.py when the GUI is in
    'Auto' mode. Returns (cli_arg, human_readable_reason).

      * BF16-capable HW (CC >= 8.0): 'auto' (let convert.py preserve source)
      * Pre-Ampere HW (CC < 8.0):    'fp16' (force all weights to F16)
      * nvidia-smi missing:          'fp16' (safe default, works everywhere)
    """
    name, cc_str = detect_nvidia_gpu()
    if name is None:
        return (
            "fp16",
            "Auto: nvidia-smi unavailable -> using --dtype fp16 (safe default).",
        )
    cc = float(cc_str)
    if cc >= BF16_MIN_COMPUTE_CAP:
        return (
            "auto",
            f"Auto: detected {name} (CC {cc_str}) -> --dtype auto "
            f"(BF16 supported; source dtype preserved).",
        )
    return (
        "fp16",
        f"Auto: detected {name} (CC {cc_str}) -> --dtype fp16 "
        f"(no native BF16; BF16-source weights will be cast to F16).",
    )


# fix_5d_tensors_{arch}.safetensors files that convert.py may leave behind.
# If they exist on a second run, ModelHyVid.handle_nd_tensor raises RuntimeError.
# We pre-empt the whole issue by folding >4D tensors before convert.py runs,
# but we also clean these stale files just in case.
# Every arch that may produce a fix_5d_tensors_*.safetensors via convert.py.
# Keep this in sync with ComfyUI-GGUF/tools/convert.py arch_list.
KNOWN_ARCH_FIX_FILES = [
    "hyvid", "wan", "hunyuan", "flux", "sd3", "aura",
    "ltxv", "hidream", "cosmos", "lumina2", "ernie",
]


class ConversionThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, src, temp_dir, out_dir,
                 dtype_cli="auto", dtype_reason="",
                 quant_type=DEFAULT_QUANT_TYPE):
        super().__init__()
        self.src = src
        self.temp_dir = temp_dir
        self.out_dir = out_dir
        # dtype_cli is one of 'auto' / 'fp16' / 'bf16' and maps directly to
        # convert.py's --dtype flag. 'auto' is the default and is a no-op
        # (convert.py preserves the source dtype).
        self.dtype_cli = dtype_cli
        self.dtype_reason = dtype_reason
        # Output quantization name passed verbatim as the third positional
        # arg to llama-quantize; e.g. 'Q4_K_M', 'Q8_0', 'F16'.
        self.quant_type = quant_type

    # ── Subprocess helper ─────────────────────────────────────────────────────

    def _stream_command(self, cmd, env):
        """Run a shell command and stream stdout/stderr line-by-line to the log."""
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
                    self.log_signal.emit(cleaned)
                buffer = ""
            else:
                buffer += char
        if buffer.strip():
            self.log_signal.emit(buffer.strip())
        return process.returncode

    # ── Step 0: Pre-process >4D tensors ──────────────────────────────────────

    def _preprocess_5d_tensors(self):
        """
        Loads the source .safetensors and checks every tensor's rank.

        Why this is needed
        ------------------
        convert.py's handle_nd_tensor() for HyVid/Wan models saves the
        offending tensor to fix_5d_tensors_{arch}.safetensors and skips it,
        so the F16 GGUF ends up missing that tensor entirely. For every other
        architecture the base-class raises NotImplementedError, aborting
        conversion outright. Either way, a 5D tensor that reaches
        llama-quantize causes GGML_ASSERT(n_dims <= GGML_MAX_DIMS) and a
        core dump.

        The fix: fold any tensor with shape (a, b, c, d, e) → (a*b, c, d, e)
        here, before convert.py ever sees the file, so handle_nd_tensor is
        never invoked.

        Returns the path to pass to convert.py (original if nothing was
        changed, a fixed copy in temp_dir otherwise).
        """
        try:
            import torch  # noqa: F401 – only needed to ensure safetensors dtype support
            from safetensors.torch import load_file, save_file
        except ImportError as exc:
            self.log_signal.emit(f"  [!] torch/safetensors not importable: {exc}")
            self.log_signal.emit("  [!] Skipping 5D scan. If quantization crashes with")
            self.log_signal.emit("      GGML_ASSERT(n_dims) check your Python environment.")
            return self.src

        self.log_signal.emit("  Scanning tensor ranks in source file…")
        tensors = load_file(self.src)

        # Heads-up about ComfyUI scaled-fp8 quantization. convert.py will
        # dequantize these in load_state_dict; we only log so the user can see
        # why the converted output is bf16 rather than fp8.
        scaled_fp8_weights = [
            k for k in tensors.keys() if k.endswith(".weight_scale")
        ]
        if scaled_fp8_weights:
            self.log_signal.emit(
                f"  Detected ComfyUI scaled-fp8 quantization on "
                f"{len(scaled_fp8_weights)} weight(s); convert.py will dequantize them."
            )

        over_4d = [k for k, v in tensors.items() if len(v.shape) > 4]
        if not over_4d:
            self.log_signal.emit("  ✓ All tensors ≤ 4D — no pre-processing needed.")
            return self.src

        self.log_signal.emit(f"  Found {len(over_4d)} tensor(s) with >4 dimensions. Folding…")

        fixed = {}
        for key, tensor in tensors.items():
            if len(tensor.shape) <= 4:
                fixed[key] = tensor
                continue
            # Fold pairs of leading dims until we reach 4D.
            # (a, b, c, d, e) → (a*b, c, d, e)
            t = tensor
            while len(t.shape) > 4:
                new_shape = (t.shape[0] * t.shape[1],) + t.shape[2:]
                t = t.reshape(new_shape)
            self.log_signal.emit(
                f"    Folded  {key}  {list(tensor.shape)}  →  {list(t.shape)}"
            )
            fixed[key] = t

        base = os.path.basename(self.src).replace(".safetensors", "_5dfixed.safetensors")
        fixed_src = os.path.join(self.temp_dir, base)
        save_file(fixed, fixed_src)
        self.log_signal.emit(f"  Fixed source written: {fixed_src}")
        return fixed_src

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def _needs_pad_fix(self, f16_path):
        """Return True iff the F16 GGUF at ``f16_path`` has a 1-D
        ``x_pad_token`` or ``cap_pad_token``. For all other models
        ``fix_pad.py`` is a no-op and we skip it to avoid an expensive
        and occasionally lossy round-trip through GGUFReader/Writer.
        """
        try:
            reader = GGUFReader(f16_path, "r")
            for t in reader.tensors:
                if t.name in ("x_pad_token", "cap_pad_token") and len(t.shape) == 1:
                    return True
            return False
        except Exception as exc:
            # Be safe: if we can't read the file, let fix_pad try.
            self.log_signal.emit(
                f"  (could not inspect {f16_path} for pad tokens: {exc}; "
                f"running fix_pad as a precaution)"
            )
            return True

    def run(self):
        filename = os.path.basename(self.src)

        my_env = os.environ.copy()
        # Absolute paths so llama-quantize finds libggml.so / libllama.so
        # regardless of which directory the GUI was launched from.
        my_env["LD_LIBRARY_PATH"] = (
            LD_PATH_BUILD_SRC + os.pathsep
            + LD_PATH_BUILD_GGML_SRC + os.pathsep
            + my_env.get("LD_LIBRARY_PATH", "")
        )
        # macOS uses DYLD_LIBRARY_PATH; set both so the same code path works.
        my_env["DYLD_LIBRARY_PATH"] = (
            LD_PATH_BUILD_SRC + os.pathsep
            + LD_PATH_BUILD_GGML_SRC + os.pathsep
            + my_env.get("DYLD_LIBRARY_PATH", "")
        )

        # Paths are always derived from the *original* filename so that the
        # F16 cache remains valid across runs even when a fixed-source copy
        # was created.
        f16_tmp   = os.path.join(self.temp_dir, filename + "_f16.gguf")
        fixed_tmp = os.path.join(self.temp_dir, filename + "_f16_fixed.gguf")
        final_out = os.path.join(
            self.out_dir,
            filename.replace(".safetensors", f"_{self.quant_type}.gguf"),
        )

        try:
            # ── Step 0: Fold >4D tensors before conversion ────────────────
            self.log_signal.emit("=== Step 0: High-dimensional tensor check ===")
            src_for_convert = self._preprocess_5d_tensors()

            # Remove any stale fix files left by a previous convert.py run.
            # ModelHyVid.handle_nd_tensor raises RuntimeError if the file
            # already exists, which would abort Step 1 on a re-run.
            for arch in KNOWN_ARCH_FIX_FILES:
                stale = f"./fix_5d_tensors_{arch}.safetensors"
                if os.path.exists(stale):
                    os.remove(stale)
                    self.log_signal.emit(f"  Removed stale fix file: {stale}")

            # ── Step 1: Convert to F16 GGUF ───────────────────────────────
            skip_convert = False
            if os.path.exists(f16_tmp):
                self.log_signal.emit(f"\n=== Step 1: Verifying cached F16 file ===")
                self.log_signal.emit(f"  Found: {f16_tmp}")
                try:
                    reader = GGUFReader(f16_tmp, "r")
                    if len(reader.tensors) > 0:
                        self.log_signal.emit(
                            f"  ✓ Valid ({len(reader.tensors)} tensors). Skipping conversion."
                        )
                        skip_convert = True
                    else:
                        self.log_signal.emit("  ✗ Empty tensor list — treating as corrupt. Reconverting.")
                except Exception as exc:
                    self.log_signal.emit(f"  ✗ Cannot read file ({exc}). Reconverting.")

            if not skip_convert:
                self.log_signal.emit("\n=== Step 1: Converting to F16 GGUF ===")
                if self.dtype_reason:
                    self.log_signal.emit(f"  {self.dtype_reason}")
                dtype_arg = (
                    "" if self.dtype_cli == "auto"
                    else f" --dtype {self.dtype_cli}"
                )
                cmd = (
                    f"{_shell_quote(sys.executable)} {_shell_quote(CONVERT_PY)} "
                    f"--src {_shell_quote(src_for_convert)} "
                    f"--dst {_shell_quote(f16_tmp)}{dtype_arg}"
                )
                if self._stream_command(cmd, my_env) != 0:
                    raise RuntimeError("F16 conversion failed. See log above.")

            # ── Step 2: Padding shape fix ─────────────────────────────────
            # The fix only does anything for Z-Image / Lumina2 models whose
            # ``x_pad_token`` / ``cap_pad_token`` arrived as 1-D tensors.
            # Newer Z-Image checkpoints (e.g. Z-Image 0.36 non-Turbo) already
            # ship them as ``[1, dim]`` -- we skip the rewrite in that case
            # to avoid an unnecessary 10+ GiB round-trip through GGUF and
            # the rare "size = 0.000 MB" corruption that triggered the
            # llama-quantize ``basic_ios::clear: iostream error``.
            self.log_signal.emit("\n=== Step 2: Applying padding shape fix ===")
            needs_fix = self._needs_pad_fix(f16_tmp)
            if needs_fix:
                cmd = (
                    f"{_shell_quote(sys.executable)} {_shell_quote(FIX_PAD_PY)} "
                    f"{_shell_quote(f16_tmp)}"
                )
                if self._stream_command(cmd, my_env) != 0:
                    raise RuntimeError("fix_pad.py failed. See log above.")
                quantize_input = fixed_tmp
            else:
                self.log_signal.emit(
                    "  No 1-D pad tokens detected -- skipping fix_pad.py. "
                    "Quantizing the F16 file directly."
                )
                quantize_input = f16_tmp

            # ── Step 3: Quantize to user-selected type ────────────────────
            self.log_signal.emit(
                f"\n=== Step 3: Quantizing to {self.quant_type} ==="
            )
            quantize_bin = (
                LLAMA_QUANTIZE_BIN_WIN
                if os.name == "nt" and os.path.exists(LLAMA_QUANTIZE_BIN_WIN)
                else LLAMA_QUANTIZE_BIN
            )
            if not os.path.exists(quantize_bin):
                raise RuntimeError(
                    f"llama-quantize binary not found at {quantize_bin}.\n"
                    "Build it first (see tools/README.md \u00a7 3), or set the\n"
                    "LLAMA_CPP_DIR environment variable to the root of your\n"
                    "llama.cpp clone."
                )
            cmd = (
                f"{_shell_quote(quantize_bin)} "
                f"{_shell_quote(quantize_input)} {_shell_quote(final_out)} "
                f"{self.quant_type}"
            )
            if self._stream_command(cmd, my_env) != 0:
                raise RuntimeError(
                    "llama-quantize failed.\n\n"
                    "If the log shows  GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS):\n"
                    "  A tensor with >4 dimensions reached the quantizer via the cached F16\n"
                    "  file (built before this fix was in place).\n\n"
                    "  Solution: delete the stale .gguf files from your temp folder\n"
                    "  and run again. Step 0 will fold those tensors before Step 1 runs."
                )

            # ── Step 4: Cleanup ───────────────────────────────────────────
            self.log_signal.emit("\n=== Step 4: Cleaning up temp files ===")
            for tmp in [f16_tmp, fixed_tmp]:
                if os.path.exists(tmp):
                    os.remove(tmp)
                    self.log_signal.emit(f"  Removed: {tmp}")

            # Also remove the _5dfixed.safetensors we may have created.
            fixed_src_path = os.path.join(
                self.temp_dir,
                filename.replace(".safetensors", "_5dfixed.safetensors"),
            )
            if os.path.exists(fixed_src_path):
                os.remove(fixed_src_path)
                self.log_signal.emit(f"  Removed: {fixed_src_path}")

            # ── Verify output ─────────────────────────────────────────────
            try:
                reader = GGUFReader(final_out)
                self.log_signal.emit(
                    f"\n✓ Output verified: {len(reader.tensors)} tensors written successfully."
                )
            except Exception as exc:
                self.log_signal.emit(f"\n⚠ Output verification warning: {exc}")

            self.finished_signal.emit(True, f"Done!\nOutput: {final_out}")

        except Exception as exc:
            self.finished_signal.emit(False, str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GGUF Converter")
        self.setAcceptDrops(True)
        self.resize(750, 650)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        # ── Source ────────────────────────────────────────────────────────────
        layout.addWidget(QLabel("Input Model (.safetensors):"))
        row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Drag & drop a .safetensors file here, or browse…")
        row.addWidget(self.input_field)
        b = QPushButton("Browse")
        b.clicked.connect(self.browse_src)
        row.addWidget(b)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setToolTip(
            "Read the model's tensor index (.safetensors header or GGUF "
            "tensor list) and recommend a quant type for the detected GPU. "
            "No weights are loaded; runs in seconds."
        )
        self.analyze_btn.clicked.connect(self.start_analyze)
        row.addWidget(self.analyze_btn)
        layout.addLayout(row)

        # ── Temp dir ──────────────────────────────────────────────────────────
        layout.addWidget(QLabel("Temporary Directory:"))
        row = QHBoxLayout()
        self.temp_field = QLineEdit()
        row.addWidget(self.temp_field)
        b = QPushButton("Browse")
        b.clicked.connect(self.browse_temp)
        row.addWidget(b)
        layout.addLayout(row)

        # ── Output dir ────────────────────────────────────────────────────────
        layout.addWidget(QLabel("Output Directory:"))
        row = QHBoxLayout()
        self.out_field = QLineEdit()
        row.addWidget(self.out_field)
        b = QPushButton("Browse")
        b.clicked.connect(self.browse_out)
        row.addWidget(b)
        layout.addLayout(row)

        # ── Output dtype selector ─────────────────────────────────────────────
        # 'Auto' probes nvidia-smi at startup and uses --dtype fp16 on Turing
        # (CC < 8.0) or pre-Ampere HW. 'Force F16' / 'Force BF16' are manual
        # overrides for debugging or for users who know the target HW better
        # than the probe does.
        self._detected_gpu, self._detected_cc = detect_nvidia_gpu()
        self._detected_vram_gb = detect_nvidia_vram_gb()
        layout.addWidget(QLabel("Output dtype:"))
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItem("Auto (detect via nvidia-smi)", "auto")
        self.dtype_combo.addItem("Force F16 (debug / Turing-compatible)", "fp16")
        self.dtype_combo.addItem("Force BF16 (debug / Ampere+ only)", "bf16")
        self.dtype_combo.currentIndexChanged.connect(self._refresh_dtype_status)
        layout.addWidget(self.dtype_combo)
        self.dtype_status = QLabel()
        self.dtype_status.setWordWrap(True)
        self.dtype_status.setStyleSheet(
            "color: #95a5a6; font-style: italic; padding-left: 4px;"
        )
        layout.addWidget(self.dtype_status)

        # ── llama-quantize output type ────────────────────────────────────────
        # Drives the third positional arg passed to llama-quantize in Step 3.
        # Default is Q4_K_M, which fits comfortably on 8 GB VRAM for 6-12B
        # image-diffusion models. The full upstream list is exposed so users
        # can pick smaller (IQ*, Q2_K, Q3_K_*) or larger (Q5/Q6/Q8) quants
        # depending on their VRAM budget.
        layout.addWidget(QLabel("Quantization type:"))
        self.quant_combo = QComboBox()
        for name, _desc in LLAMA_QUANTIZE_TYPES:
            self.quant_combo.addItem(name, name)
        self.quant_combo.setCurrentIndex(
            self.quant_combo.findData(DEFAULT_QUANT_TYPE)
        )
        self.quant_combo.currentIndexChanged.connect(self._refresh_quant_status)
        layout.addWidget(self.quant_combo)
        self.quant_status = QLabel()
        self.quant_status.setWordWrap(True)
        self.quant_status.setStyleSheet(
            "color: #95a5a6; font-style: italic; padding-left: 4px;"
        )
        layout.addWidget(self.quant_status)

        # ── Save preset ───────────────────────────────────────────────────────
        btn_save = QPushButton("Save Temp & Output Paths as Preset")
        btn_save.setStyleSheet("background-color: #34495e; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_preset)
        layout.addWidget(btn_save)

        # ── Run ───────────────────────────────────────────────────────────────
        self.run_btn = QPushButton("▶  Run GGUF Conversion Pipeline")
        self.run_btn.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; "
            "font-size: 14px; height: 35px;"
        )
        self.run_btn.clicked.connect(self.start_workflow)
        layout.addWidget(self.run_btn)

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #3498db; }")
        layout.addWidget(self.progress_bar)

        # ── Log ───────────────────────────────────────────────────────────────
        layout.addWidget(QLabel("Log:"))
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; font-family: monospace;"
        )
        layout.addWidget(self.log_area)

        self.load_preset()
        self._refresh_dtype_status()
        self._refresh_quant_status()

    # ── File browsers ─────────────────────────────────────────────────────────

    def browse_src(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "",
            "Weights (*.safetensors *.gguf);;safetensors (*.safetensors);;GGUF (*.gguf);;All files (*)",
        )
        if path:
            self.input_field.setText(path.strip())

    def browse_temp(self):
        path = QFileDialog.getExistingDirectory(self, "Select Temporary Directory")
        if path:
            self.temp_field.setText(path.strip())

    def browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.out_field.setText(path.strip())

    # ── Drag-and-drop ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        file_path = ""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile() or url.toString()
        elif event.mimeData().hasText():
            file_path = event.mimeData().text()

        file_path = file_path.strip()
        if file_path.startswith("file://"):
            file_path = file_path[7:]
        file_path = unquote(file_path).strip()

        if file_path.endswith((".safetensors", ".gguf")):
            self.input_field.setText(file_path)
            if not self.out_field.text().strip():
                self.out_field.setText(os.path.dirname(file_path))

    # ── Preset persistence ────────────────────────────────────────────────────

    def save_preset(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({
                    "temp": self.temp_field.text().strip(),
                    "output": self.out_field.text().strip(),
                    "dtype_mode": self.dtype_combo.currentData(),
                    "quant_type": self.quant_combo.currentData(),
                }, f)
            self.log_area.appendPlainText("Preset saved.")
        except Exception as exc:
            self.log_area.appendPlainText(f"Error saving preset: {exc}")

    def load_preset(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                self.temp_field.setText(data.get("temp", "").strip())
                self.out_field.setText(data.get("output", "").strip())
                mode = data.get("dtype_mode")
                if mode is None:
                    # Backward-compat with PR #2's settings.json schema.
                    mode = "fp16" if data.get("force_fp16", True) else "auto"
                idx = self.dtype_combo.findData(mode)
                if idx >= 0:
                    self.dtype_combo.setCurrentIndex(idx)
                quant_type = data.get("quant_type", DEFAULT_QUANT_TYPE)
                qidx = self.quant_combo.findData(quant_type)
                if qidx >= 0:
                    self.quant_combo.setCurrentIndex(qidx)
            except Exception as exc:
                self.log_area.appendPlainText(f"Error loading preset: {exc}")

    # ── dtype status label ────────────────────────────────────────────────────

    def _refresh_dtype_status(self):
        """Update the small italic status label below the dtype combo so the
        user can see exactly which --dtype value will be passed to convert.py.
        """
        mode = self.dtype_combo.currentData()
        if mode == "auto":
            if self._detected_gpu is None:
                self.dtype_status.setText(
                    "nvidia-smi not available -> Auto resolves to --dtype fp16 "
                    "(safe default for CPU / ROCm / Apple Silicon)."
                )
            else:
                cc = float(self._detected_cc)
                if cc >= BF16_MIN_COMPUTE_CAP:
                    self.dtype_status.setText(
                        f"Detected: {self._detected_gpu} (CC {self._detected_cc})"
                        f" -> --dtype auto (BF16 supported)."
                    )
                else:
                    self.dtype_status.setText(
                        f"Detected: {self._detected_gpu} (CC {self._detected_cc})"
                        f" -> --dtype fp16 (no native BF16 support)."
                    )
        elif mode == "fp16":
            self.dtype_status.setText(
                "Override: --dtype fp16 (every BF16-source weight is cast to F16)."
            )
        else:  # 'bf16'
            self.dtype_status.setText(
                "Override: --dtype bf16 (invalid on Turing / pre-Ampere GPUs)."
            )

    def _refresh_quant_status(self):
        """Show the upstream llama-quantize description for the selected
        type, so the user can see the rough size + perplexity tradeoff in
        the UI without consulting quantize.cpp.
        """
        name = self.quant_combo.currentData()
        desc = next((d for n, d in LLAMA_QUANTIZE_TYPES if n == name), "")
        self.quant_status.setText(f"{name}: {desc}")

    # ── Workflow ──────────────────────────────────────────────────────────────

    def start_workflow(self):
        src      = self.input_field.text().strip()
        temp_dir = self.temp_field.text().strip()
        out_dir  = self.out_field.text().strip()

        if not src or not temp_dir or not out_dir:
            self.log_area.appendPlainText("Error: all three paths must be filled in.")
            return
        if not os.path.exists(src):
            self.log_area.appendPlainText(f"Error: source file not found:\n  {src}")
            return

        self.log_area.clear()
        self.run_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)   # indeterminate while running

        mode = self.dtype_combo.currentData()
        if mode == "auto":
            dtype_cli, dtype_reason = resolve_auto_dtype()
        elif mode == "fp16":
            dtype_cli, dtype_reason = (
                "fp16",
                "Override: forcing --dtype fp16 (all BF16-source weights cast to F16).",
            )
        else:  # 'bf16'
            dtype_cli, dtype_reason = (
                "bf16",
                "Override: forcing --dtype bf16 (only valid on Ampere or newer).",
            )

        self.worker = ConversionThread(
            src, temp_dir, out_dir,
            dtype_cli=dtype_cli, dtype_reason=dtype_reason,
            quant_type=self.quant_combo.currentData(),
        )
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    @Slot(str)
    def update_log(self, text):
        self.log_area.appendPlainText(text)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self, success, msg):
        self.run_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self.log_area.appendPlainText("\n" + "=" * 40)
        self.log_area.appendPlainText(msg)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -- Analyze ---------------------------------------------------------

    def start_analyze(self):
        """Run analyze_model.analyze() against the currently-selected
        .safetensors or .gguf path on a worker thread, then pop the result
        dialog. GGUF inputs are read via gguf.GGUFReader -- useful for
        re-evaluating an already-converted intermediate against the same
        VRAM / quant matrix.
        """
        src = self.input_field.text().strip()
        if not src:
            QMessageBox.warning(
                self, "Analyze",
                "Pick a .safetensors or .gguf file first (Browse or drag-and-drop).",
            )
            return
        if not os.path.isfile(src):
            QMessageBox.warning(
                self, "Analyze", f"File not found:\n  {src}",
            )
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")

        self.analyze_worker = AnalyzeWorker(
            src,
            gpu_vram_gb=self._detected_vram_gb,
            gpu_name=self._detected_gpu,
        )
        self.analyze_worker.finished_signal.connect(self.on_analyze_finished)
        self.analyze_worker.start()

    @Slot(object, str)
    def on_analyze_finished(self, result, error):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        if error:
            QMessageBox.critical(self, "Analyze failed", error)
            return
        dlg = AnalyzeResultDialog(result, self)
        dlg.quant_chosen.connect(self.apply_quant_recommendation)
        dlg.exec()

    @Slot(str)
    def apply_quant_recommendation(self, quant_name):
        """Wire the dialog's 'Use this quant' button to the main combo so
        the user doesn't have to re-pick the type after closing Analyze.
        """
        idx = self.quant_combo.findData(quant_name)
        if idx >= 0:
            self.quant_combo.setCurrentIndex(idx)
            self.log_area.appendPlainText(
                f"Quant type set to {quant_name} from Analyze recommendation."
            )


# -- Analyze worker + dialog --------------------------------------------

class AnalyzeWorker(QThread):
    """Off-main-thread wrapper around analyze_model.analyze() so the UI
    doesn't freeze while the safetensors header is parsed. The header
    read itself is ~milliseconds, but architecture detection on a model
    with thousands of keys can take noticeable time.
    """
    finished_signal = Signal(object, str)   # (AnalysisResult or None, error str)

    def __init__(self, path, gpu_vram_gb, gpu_name):
        super().__init__()
        self.path = path
        self.gpu_vram_gb = gpu_vram_gb
        self.gpu_name = gpu_name

    def run(self):
        try:
            result = analyze_model.analyze(
                self.path,
                gpu_vram_gb=self.gpu_vram_gb,
                gpu_name=self.gpu_name,
            )
            self.finished_signal.emit(result, "")
        except Exception:
            self.finished_signal.emit(None, traceback.format_exc())


class AnalyzeResultDialog(QDialog):
    """Modal-ish dialog that renders an analyze_model.AnalysisResult.

    Layout (top to bottom):
      * Header: file path, params, arch, hidden dim, GPU/VRAM.
      * Recommendation banner (highlighted) + 'Use this quant' button.
      * Quant table (one row per quant, columns per resolution).
      * Notes (only if analyze surfaced any).
      * Formula box (always visible, read-only).
    """
    quant_chosen = Signal(str)

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("Model Analysis")
        self.resize(900, 600)
        layout = QVBoxLayout(self)

        # Header
        arch_str = result.arch or "unknown"
        klass = result.arch_class_name or "-"
        head_text = (
            f"<b>File:</b> {os.path.basename(result.path)} "
            f"({analyze_model.fmt_bytes(result.file_size)})<br>"
            f"<b>Arch:</b> {arch_str} ({klass})"
            + ("  &mdash; <i>INVALID variant</i>" if result.invalid else "")
            + f"<br><b>Params:</b> {analyze_model.fmt_params(result.n_params)} "
            f"across {result.n_tensors} tensors<br>"
            f"<b>hidden_dim:</b> {result.dims.hidden_dim or '?'} &nbsp; "
            f"<b>num_layers:</b> {result.dims.num_layers or '?'} &nbsp; "
            f"<b>patch_size:</b> {result.dims.patch_size or '?'} &nbsp; "
            f"<b>in_channels:</b> {result.dims.in_channels or '?'}<br>"
        )
        if result.gpu_vram_gb is not None:
            head_text += (
                f"<b>GPU:</b> {result.gpu_name or '?'} "
                f"({result.gpu_vram_gb:.1f} GB) &nbsp; "
                f"<b>headroom:</b> {result.headroom_gb:.1f} GB &nbsp; "
                f"<b>overhead:</b> {result.comfyui_overhead_mb} MB"
            )
        else:
            head_text += (
                "<i>GPU not detected (nvidia-smi unavailable) -- fits/recommendation "
                "columns disabled.</i>"
            )
        header = QLabel(head_text)
        header.setWordWrap(True)
        layout.addWidget(header)

        # Recommendation banner
        rec_row = QHBoxLayout()
        if result.recommended:
            rec_label = QLabel(
                f"<b>Recommended quant:</b> {result.recommended}  "
                f"(highest-quality option that fits with {result.headroom_gb:.0f} GB headroom "
                f"at the largest configured resolution)"
            )
            rec_label.setStyleSheet(
                "background-color: #27ae60; color: white; "
                "padding: 8px; border-radius: 4px;"
            )
            use_btn = QPushButton(f"Use {result.recommended}")
            use_btn.clicked.connect(lambda: self._emit_chosen(result.recommended))
        else:
            why = (
                "no GPU detected (run on a machine with nvidia-smi or pass --vram)."
                if result.gpu_vram_gb is None
                else "every modelled quant exceeds the budget; lower the resolution or free VRAM."
            )
            rec_label = QLabel(f"<b>No recommendation</b> -- {why}")
            rec_label.setStyleSheet(
                "background-color: #c0392b; color: white; "
                "padding: 8px; border-radius: 4px;"
            )
            use_btn = None
        rec_label.setWordWrap(True)
        rec_row.addWidget(rec_label, stretch=1)
        if use_btn is not None:
            rec_row.addWidget(use_btn)
        layout.addLayout(rec_row)

        # Results table
        headers = ["Quant", "Weight"]
        for label, _, _ in result.resolutions:
            headers += [f"Act@{label}", f"Total@{label}", f"Fits@{label}"]
        table = QTableWidget(len(result.rows), len(headers), self)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setToolTip(
            "Weight: per-tensor weight cost after applying convert.py's "
            "keep-F32 rules + ggml bpw table.\n"
            "Act: SDPA / flash-attention activation peak at the listed "
            "PIXEL resolution (VAE 8x downsample applied).\n"
            "Total = Weight + Act + 400 MB ComfyUI overhead.\n"
            "Fits = Total <= (VRAM - 1 GB headroom)."
        )

        for r, row in enumerate(result.rows):
            items = [QTableWidgetItem(row.name),
                     QTableWidgetItem(analyze_model.fmt_bytes(row.weight_bytes))]
            for label, _, _ in result.resolutions:
                items.append(QTableWidgetItem(
                    analyze_model.fmt_bytes(row.activations_bytes[label])))
                items.append(QTableWidgetItem(
                    analyze_model.fmt_bytes(row.total_bytes[label])))
                fit_item = QTableWidgetItem(
                    "YES" if row.fits[label]
                    else ("-" if result.gpu_vram_gb is None else "no")
                )
                if row.fits[label]:
                    fit_item.setForeground(QColor("#27ae60"))
                elif result.gpu_vram_gb is not None:
                    fit_item.setForeground(QColor("#c0392b"))
                items.append(fit_item)
            if result.recommended == row.name:
                font = QFont()
                font.setBold(True)
                for it in items:
                    it.setFont(font)
                    it.setBackground(QColor("#1e8449"))
                    it.setForeground(QColor("white"))
            for c, it in enumerate(items):
                table.setItem(r, c, it)
        layout.addWidget(table, stretch=1)

        # Notes (only if any)
        if result.notes:
            notes_label = QLabel("<b>Notes:</b>")
            layout.addWidget(notes_label)
            notes_text = QTextEdit()
            notes_text.setReadOnly(True)
            notes_text.setMaximumHeight(80)
            notes_text.setPlainText("\n".join("* " + n for n in result.notes))
            layout.addWidget(notes_text)

        # Formula (auditable)
        layout.addWidget(QLabel("<b>Formula (auditable; 100% model-derived):</b>"))
        formula = QTextEdit()
        formula.setReadOnly(True)
        formula.setMaximumHeight(110)
        formula.setStyleSheet("font-family: monospace;")
        formula.setPlainText(result.activation_formula)
        layout.addWidget(formula)

        # Close
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        layout.addWidget(bb)

    def _emit_chosen(self, quant_name):
        self.quant_chosen.emit(quant_name)
        self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
