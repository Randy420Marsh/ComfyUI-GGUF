import sys
import json
import os
import subprocess
from urllib.parse import unquote
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QLineEdit, QPushButton, QLabel, QFileDialog,
                               QPlainTextEdit, QHBoxLayout, QProgressBar,
                               QCheckBox)
from PySide6.QtCore import QThread, Signal, Slot
from gguf import GGUFReader

CONFIG_FILE = "settings.json"

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

    def __init__(self, src, temp_dir, out_dir, force_fp16=False):
        super().__init__()
        self.src = src
        self.temp_dir = temp_dir
        self.out_dir = out_dir
        self.force_fp16 = force_fp16

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

    def run(self):
        filename = os.path.basename(self.src)

        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = (
            "./llama.cpp/build/src:./llama.cpp/build/ggml/src:"
            + my_env.get("LD_LIBRARY_PATH", "")
        )

        # Paths are always derived from the *original* filename so that the
        # F16 cache remains valid across runs even when a fixed-source copy
        # was created.
        f16_tmp   = os.path.join(self.temp_dir, filename + "_f16.gguf")
        fixed_tmp = os.path.join(self.temp_dir, filename + "_f16_fixed.gguf")
        final_out = os.path.join(
            self.out_dir,
            filename.replace(".safetensors", "_Q4_K_M.gguf"),
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
                dtype_arg = " --dtype fp16" if self.force_fp16 else ""
                if self.force_fp16:
                    self.log_signal.emit(
                        "  Forcing --dtype fp16 (RTX 20xx / no native bf16): "
                        "bf16-source weights will be written as F16 in the GGUF."
                    )
                cmd = (
                    f"python ComfyUI-GGUF/tools/convert.py "
                    f"--src '{src_for_convert}' --dst '{f16_tmp}'{dtype_arg}"
                )
                if self._stream_command(cmd, my_env) != 0:
                    raise RuntimeError("F16 conversion failed. See log above.")

            # ── Step 2: Padding shape fix ─────────────────────────────────
            self.log_signal.emit("\n=== Step 2: Applying padding shape fix ===")
            cmd = f"python ComfyUI-GGUF/tools/fix_pad.py '{f16_tmp}'"
            if self._stream_command(cmd, my_env) != 0:
                raise RuntimeError("fix_pad.py failed. See log above.")

            # ── Step 3: Quantize to Q4_K_M ───────────────────────────────
            self.log_signal.emit("\n=== Step 3: Quantizing to Q4_K_M ===")
            cmd = f"./llama.cpp/build/bin/llama-quantize '{fixed_tmp}' '{final_out}' Q4_K_M"
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

        # ── Force fp16 toggle ─────────────────────────────────────────────────
        # Turing (RTX 20xx) and earlier GPUs have no native bf16 — bf16 tensors
        # get upcast to fp32 at inference, doubling memory and tanking throughput.
        # Default ON because the typical user of this fork runs on a 2070S.
        self.force_fp16_box = QCheckBox(
            "Force F16 output (no BF16 — required for RTX 20xx / Turing)"
        )
        self.force_fp16_box.setChecked(True)
        layout.addWidget(self.force_fp16_box)

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

    # ── File browsers ─────────────────────────────────────────────────────────

    def browse_src(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "Weights (*.safetensors)"
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

        if file_path.endswith(".safetensors"):
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
                    "force_fp16": self.force_fp16_box.isChecked(),
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
                # Default to True so first-time users on RTX 20xx aren't bitten
                # by the bf16-upcast issue.
                self.force_fp16_box.setChecked(bool(data.get("force_fp16", True)))
            except Exception as exc:
                self.log_area.appendPlainText(f"Error loading preset: {exc}")

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

        self.worker = ConversionThread(
            src, temp_dir, out_dir,
            force_fp16=self.force_fp16_box.isChecked(),
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
