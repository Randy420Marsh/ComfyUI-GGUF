"""Smoke tests for tools/pipeline_lib.py.

These tests do not require torch / safetensors / PySide6 / a real
llama-quantize build — they only exercise the pure-Python helpers
(``locate_llama_quantize``, ``shell_quote``, ``build_subprocess_env``,
``needs_pad_fix``, ``get_gguf_arch``) plus the CLI argparse layer.

The actual ``run_pipeline`` integration is covered by manual end-to-end
testing on a workstation with a GPU + llama-quantize binary; replicating
that on CI would need ~12 GiB of safetensors fixtures.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import pipeline_lib  # noqa: E402
import gguf_pipeline  # noqa: E402


class LocateLlamaQuantizeTests(unittest.TestCase):
    """Pre-flight binary detection."""

    def test_missing_binary_returns_helpful_error(self):
        with mock.patch.object(pipeline_lib.os.path, "exists", return_value=False):
            path, err = pipeline_lib.locate_llama_quantize()
        self.assertIsNotNone(err)
        self.assertIn("llama-quantize binary not found", err)
        self.assertIn("Build it first", err)
        self.assertTrue(
            "city96" in err,
            "error should reference the pre-patched fork",
        )
        self.assertTrue(path.endswith("llama-quantize") or path.endswith(".exe"))

    def test_override_env_var_appears_in_error(self):
        with mock.patch.dict(os.environ, {"LLAMA_CPP_DIR": "/nonexistent/path"}):
            with mock.patch.object(pipeline_lib.os.path, "exists", return_value=False):
                # Re-resolve under the override.
                resolved = pipeline_lib._resolve_llama_cpp_dir()
            self.assertEqual(resolved, "/nonexistent/path")

    def test_found_binary_returns_no_error(self):
        with mock.patch.object(pipeline_lib.os.path, "exists", return_value=True):
            path, err = pipeline_lib.locate_llama_quantize()
        self.assertIsNone(err)
        self.assertTrue(path.endswith("llama-quantize") or path.endswith(".exe"))


class ShellQuoteTests(unittest.TestCase):
    """Argument quoting roundtrips."""

    def test_posix_quotes_path_with_spaces(self):
        with mock.patch.object(pipeline_lib.os, "name", "posix"):
            q = pipeline_lib.shell_quote("/path with spaces/file.gguf")
        self.assertEqual(q, "'/path with spaces/file.gguf'")

    def test_posix_escapes_single_quote(self):
        with mock.patch.object(pipeline_lib.os, "name", "posix"):
            q = pipeline_lib.shell_quote("/tmp/it's_a_file.gguf")
        # Inside a single-quoted POSIX string, ' is escaped as '\''.
        self.assertEqual(q, "'/tmp/it'\\''s_a_file.gguf'")

    def test_windows_uses_double_quotes(self):
        with mock.patch.object(pipeline_lib.os, "name", "nt"):
            q = pipeline_lib.shell_quote(r"C:\Program Files\file.gguf")
        self.assertEqual(q, r'"C:\Program Files\file.gguf"')


class BuildSubprocessEnvTests(unittest.TestCase):
    """LD_LIBRARY_PATH / DYLD_LIBRARY_PATH injection."""

    def test_prepends_build_paths(self):
        env = pipeline_lib.build_subprocess_env()
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith(
            pipeline_lib.LD_PATH_BUILD_SRC
        ))
        self.assertIn(pipeline_lib.LD_PATH_BUILD_GGML_SRC, env["LD_LIBRARY_PATH"])
        self.assertIn(pipeline_lib.LD_PATH_BUILD_GGML_SRC, env["DYLD_LIBRARY_PATH"])

    def test_preserves_existing_ld_path(self):
        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/opt/cuda/lib64"}):
            env = pipeline_lib.build_subprocess_env()
        self.assertIn("/opt/cuda/lib64", env["LD_LIBRARY_PATH"])


# ── Synthetic GGUF helper, copy of the one in test_analyze_model_gguf_shape ──
# Re-implemented locally so this test file remains independent of the other.

def _write_synthetic_gguf(path: str, tensors: list[tuple[str, list[int], str]],
                          arch: str = "lumina2") -> None:
    """Write a stripped-down GGUF v3 file with the given tensor metadata.

    Uses :class:`gguf.GGUFWriter` for correctness (we trust the library
    to produce a well-formed file even though we only need the header /
    tensor index, not the actual tensor data).
    """
    import gguf
    import numpy as np
    w = gguf.GGUFWriter(path=path, arch=arch)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16))
    for name, shape, dtype in tensors:
        if dtype == "F16":
            data = np.zeros(shape, dtype=np.float16)
        elif dtype == "F32":
            data = np.zeros(shape, dtype=np.float32)
        else:
            raise ValueError(f"unknown synthetic dtype {dtype}")
        w.add_tensor(name, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class NeedsPadFixTests(unittest.TestCase):
    """Step-2 skip heuristic."""

    def test_returns_true_when_1d_pad_token_present(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "lumina_1d_pad.gguf")
            _write_synthetic_gguf(p, [
                ("x_pad_token", [3840], "F16"),     # 1-D \u2192 legacy Z-Image
                ("cap_embedder.1.weight", [1, 1], "F16"),
            ])
            self.assertTrue(pipeline_lib.needs_pad_fix(p))

    def test_returns_false_when_pad_tokens_already_2d(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "lumina_2d_pad.gguf")
            _write_synthetic_gguf(p, [
                ("x_pad_token", [1, 3840], "F16"),  # 2-D \u2192 Z-Image 0.36
                ("cap_pad_token", [1, 3840], "F16"),
            ])
            self.assertFalse(pipeline_lib.needs_pad_fix(p))

    def test_returns_false_when_no_pad_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "flux.gguf")
            _write_synthetic_gguf(p, [
                ("img_in.weight", [3072, 64], "F16"),
            ], arch="flux")
            self.assertFalse(pipeline_lib.needs_pad_fix(p))

    def test_returns_true_when_file_unreadable(self):
        bad = "/nonexistent/path/missing.gguf"
        # Defensive: should opt to run fix_pad rather than silently skip.
        self.assertTrue(pipeline_lib.needs_pad_fix(bad))


class GgufArchTests(unittest.TestCase):
    """``get_gguf_arch`` round-trip."""

    def test_reads_lumina2_arch_back(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model.gguf")
            _write_synthetic_gguf(p, [("dummy", [1], "F16")], arch="lumina2")
            self.assertEqual(pipeline_lib.get_gguf_arch(p), "lumina2")

    def test_reads_hyvid_arch_back(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model.gguf")
            _write_synthetic_gguf(p, [("dummy", [1], "F16")], arch="hyvid")
            self.assertEqual(pipeline_lib.get_gguf_arch(p), "hyvid")

    def test_returns_none_on_missing_file(self):
        self.assertIsNone(pipeline_lib.get_gguf_arch("/nonexistent/path.gguf"))


class CliArgparseTests(unittest.TestCase):
    """Verify ``gguf_pipeline.py`` exposes all of pipeline_lib's quant types."""

    def test_quant_choices_match_pipeline_lib(self):
        # Parsing --quant with each supported type must succeed.
        for name in pipeline_lib.QUANTIZE_TYPE_NAMES:
            ns = gguf_pipeline.parse_args(
                ["--src", "x.safetensors", "--dst-dir", "/out", "--quant", name]
            )
            self.assertEqual(ns.quant, name)

    def test_unknown_quant_rejected(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):
                gguf_pipeline.parse_args(
                    ["--src", "x", "--dst-dir", "/o", "--quant", "Q99_BOGUS"]
                )

    def test_dtype_choices(self):
        for d in ("auto", "fp16", "bf16"):
            ns = gguf_pipeline.parse_args(
                ["--src", "x", "--dst-dir", "/o", "--dtype", d]
            )
            self.assertEqual(ns.dtype, d)

    def test_default_quant_is_q4_k_m(self):
        ns = gguf_pipeline.parse_args(["--src", "x", "--dst-dir", "/o"])
        self.assertEqual(ns.quant, "Q4_K_M")
        self.assertEqual(ns.dtype, "auto")
        self.assertFalse(ns.keep_intermediate)
        self.assertFalse(ns.no_fix_5d)

    def test_help_lists_every_quant_type(self):
        # --help triggers SystemExit(0) after printing usage to stdout.
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                gguf_pipeline.parse_args(["--help"])
        text = buf.getvalue()
        for name, _desc in pipeline_lib.LLAMA_QUANTIZE_TYPES:
            self.assertIn(name, text, f"--help should mention {name}")


class PipelineLibConstantsTests(unittest.TestCase):
    """Sanity: constants exposed for GUI re-export must exist and be valid."""

    def test_default_quant_is_known(self):
        self.assertIn(
            pipeline_lib.DEFAULT_QUANT_TYPE,
            pipeline_lib.QUANTIZE_TYPE_NAMES,
        )

    def test_known_arch_fix_files_subset_of_convert_arch_list(self):
        # KNOWN_ARCH_FIX_FILES must be a subset of arches that convert.py
        # actually knows about, otherwise we'd never clean their stale files.
        from convert import arch_list
        convert_archs = {cls.arch for cls in arch_list}
        for arch in pipeline_lib.KNOWN_ARCH_FIX_FILES:
            self.assertIn(arch, convert_archs,
                          f"KNOWN_ARCH_FIX_FILES has {arch!r} which is not in"
                          f" convert.py arch_list")

    def test_archs_needing_5d_reattach_subset(self):
        for arch in pipeline_lib.ARCHS_NEEDING_5D_REATTACH:
            self.assertIn(arch, pipeline_lib.KNOWN_ARCH_FIX_FILES)

    def test_known_arch_fix_files_covers_convert_arch_list(self):
        # The inverse of the subset test: every arch convert.py can emit
        # must have its potential stale fix file cleaned up by Step 0.
        from convert import arch_list
        for cls in arch_list:
            self.assertIn(cls.arch, pipeline_lib.KNOWN_ARCH_FIX_FILES,
                          f"convert.py arch {cls.arch!r} missing from"
                          f" KNOWN_ARCH_FIX_FILES")


class InsertNameSuffixTests(unittest.TestCase):
    """Output naming must only touch the final extension."""

    def test_plain_safetensors(self):
        self.assertEqual(
            pipeline_lib.insert_name_suffix("model.safetensors", "_5dfixed"),
            "model_5dfixed.safetensors",
        )

    def test_sft_extension_gets_distinct_name(self):
        # A bare str.replace(".safetensors", ...) would no-op here.
        self.assertEqual(
            pipeline_lib.insert_name_suffix("model.sft", "_5dfixed"),
            "model_5dfixed.sft",
        )

    def test_double_extension_only_touches_last(self):
        self.assertEqual(
            pipeline_lib.insert_name_suffix("model.safetensors_f16.gguf", "_fixed"),
            "model.safetensors_f16_fixed.gguf",
        )

    def test_no_extension(self):
        self.assertEqual(
            pipeline_lib.insert_name_suffix("model", "_5dfixed"),
            "model_5dfixed",
        )


class StreamCommandTests(unittest.TestCase):
    """Subprocess streaming: line splitting, CR handling, return codes."""

    def _run(self, code: str) -> tuple[int, list[str]]:
        lines = []
        cmd = (
            f"{pipeline_lib.shell_quote(sys.executable)} -c "
            f"{pipeline_lib.shell_quote(code)}"
        )
        rc = pipeline_lib.stream_command(cmd, os.environ.copy(), lines.append)
        return rc, lines

    def test_streams_lines_and_returns_zero(self):
        rc, lines = self._run("print('alpha'); print('beta')")
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["alpha", "beta"])

    def test_carriage_return_progress_lines_split(self):
        rc, lines = self._run(r"import sys; sys.stdout.write('10%\r20%\n')")
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["10%", "20%"])

    def test_nonzero_exit_code_propagates(self):
        rc, _lines = self._run("import sys; sys.exit(3)")
        self.assertEqual(rc, 3)

    def test_stdout_closed_before_exit_does_not_hang(self):
        # Child closes stdout, then keeps running briefly: stream_command
        # must wait for the real exit code instead of spinning or hanging.
        rc, lines = self._run(
            "import os, time; print('done', flush=True);"
            " os.close(1); time.sleep(0.2)"
        )
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["done"])


if __name__ == "__main__":
    unittest.main()
