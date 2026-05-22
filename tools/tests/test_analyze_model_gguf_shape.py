"""Regression test for the GGUF shape-order bug in ``analyze_model``.

Background: GGUF stores tensor dimensions in *reversed* on-disk order
relative to numpy / torch / safetensors. ``GGUFWriter.write_ti_data_to_file``
packs ``ti.shape[n_dims - 1 - j]`` and ``GGUFReader._build_tensors``
leaves the ``shape`` field in that reversed form (only ``.data`` is
reshaped to numpy order via ``np_dims = tuple(reversed(dims.tolist()))``).

Before the fix, ``analyze_model.read_gguf_tensors`` returned the reversed
on-disk shape directly, so for a 4-D patch-embedding conv
``(out_channels=1280, in_channels=4, kh=2, kw=2)`` (torch convention) the
analyzer saw ``(2, 2, 4, 1280)`` from a GGUF input and ``(1280, 4, 2, 2)``
from a safetensors input. ``extract_dims`` then read
``hidden_dim = shape[0]`` and arrived at completely different answers for
the two formats (``hidden_dim = 2`` vs ``1280``).

This test rebuilds the same logical model twice — once as a synthetic
``.safetensors`` and once as a synthetic ``.gguf`` — and asserts that
``read_model_tensors`` produces shape-equivalent ``TensorInfo`` lists.

Run with: ``python -m unittest tools.tests.test_analyze_model_gguf_shape``
or from the repo root: ``python -m unittest discover -s tools/tests``.
"""

import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

# Make ``tools/`` importable regardless of where the test runner is invoked
# from.  Mirrors what users do interactively: ``python tools/analyze_model.py``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.dirname(_HERE)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import analyze_model  # noqa: E402  (sys.path manipulation above)


# A handful of distinctively-shaped tensors chosen so each axis is a
# unique value — this way any shape-order regression is immediately
# obvious from the diff rather than silently passing because two axes
# happen to share a size.
_TENSORS = {
    # 4-D patch-embedding conv (out, in, kh, kw) — the classic case the
    # Devin Review caught.  Each axis is distinct.
    "x_embedder.proj.weight": ("F16", (1280, 4, 2, 2)),
    # 2-D Linear weight (out, in) — Linear(2560, 3840) in torch.
    "cap_embedder.1.weight":  ("F16", (3840, 2560)),
    # 2-D Linear weight for arch detection (the second half of the
    # ModelLumina2.keys_detect tuple).  qkv = 3 * hidden in lumina2.
    "context_refiner.0.attention.qkv.weight": ("F16", (11520, 3840)),
    # 1-D bias / norm — should round-trip unchanged (reversed of a 1-D
    # tuple is itself, so this exercises the no-op case explicitly).
    "cap_embedder.1.bias":    ("F32", (3840,)),
    # 2-D Linear in a numbered layer block, so ``extract_dims`` picks
    # up a layer count too.
    "layers.0.feed_forward.w1.weight": ("F16", (10240, 3840)),
}


# Map ``analyze_model`` dtype names to (numpy_dtype, bytes_per_element).
_DTYPE_TABLE = {
    "F16":  (np.float16,  2),
    "F32":  (np.float32,  4),
    "BF16": (np.uint16,   2),  # bf16 has no numpy dtype; emulate as u16.
}


def _write_synthetic_safetensors(path: str) -> None:
    """Build a safetensors file whose header advertises ``_TENSORS``.

    The body is zero-filled — ``analyze_model`` never reads tensor data,
    only the header.  This keeps the test fast and avoids depending on
    ``safetensors`` as a runtime dep of the tests.
    """
    header = {}
    offset = 0
    for name, (dtype, shape) in _TENSORS.items():
        nbytes = _DTYPE_TABLE[dtype][1]
        for d in shape:
            nbytes *= d
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    hdr_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(b"\x00" * offset)


def _write_synthetic_gguf(path: str) -> None:
    """Build a real GGUF file via ``gguf.GGUFWriter`` containing ``_TENSORS``.

    Uses the actual library so we exercise the same writer the production
    pipeline (``convert.py`` + ``fix_pad.py``) uses, which is what gives
    the test its teeth: any future change to the dim packing convention
    in either ``GGUFWriter`` or ``GGUFReader`` will trip this test.
    """
    import gguf as _gguf
    w = _gguf.GGUFWriter(path, arch="lumina2")
    for name, (dtype, shape) in _TENSORS.items():
        np_dtype, _ = _DTYPE_TABLE[dtype]
        # ``add_tensor`` accepts a torch-/numpy-ordered shape and is the
        # entry point ``convert.py`` itself uses, so this exercises the
        # same code path the real pipeline does.
        w.add_tensor(name, np.zeros(shape, dtype=np_dtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class GGUFShapeOrderRegressionTest(unittest.TestCase):
    """Asserts ``read_gguf_tensors`` reports torch-/safetensors-compatible shapes.

    Before the fix this test would fail on every 2-D and 4-D tensor (the
    reversed dims happen to be a no-op only for 1-D tensors).
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="gguf-shape-test-")
        self.gguf_path = os.path.join(self.tmpdir, "synth.gguf")
        self.st_path = os.path.join(self.tmpdir, "synth.safetensors")
        _write_synthetic_gguf(self.gguf_path)
        _write_synthetic_safetensors(self.st_path)

    def test_detect_format_routes_correctly(self):
        self.assertEqual(analyze_model._detect_format(self.gguf_path), "gguf")
        self.assertEqual(analyze_model._detect_format(self.st_path), "safetensors")

    def test_shapes_match_between_formats(self):
        gguf_tensors, _ = analyze_model.read_model_tensors(self.gguf_path)
        st_tensors, _ = analyze_model.read_model_tensors(self.st_path)
        gguf_by_name = {t.name: t for t in gguf_tensors}
        st_by_name = {t.name: t for t in st_tensors}
        self.assertEqual(set(gguf_by_name), set(st_by_name))
        for name, st in st_by_name.items():
            g = gguf_by_name[name]
            self.assertEqual(
                g.shape, st.shape,
                f"shape mismatch on {name!r}: gguf={g.shape!r} st={st.shape!r}",
            )
            self.assertEqual(g.n_params, st.n_params, name)
            self.assertEqual(g.dtype, st.dtype, name)

    def test_extract_dims_match_between_formats(self):
        gguf_tensors, _ = analyze_model.read_model_tensors(self.gguf_path)
        st_tensors, _ = analyze_model.read_model_tensors(self.st_path)
        g_dims = analyze_model.extract_dims(gguf_tensors)
        st_dims = analyze_model.extract_dims(st_tensors)

        # Both paths should pick the same patch-embedder key as the
        # hidden_dim source (``x_embedder.proj.weight`` is the highest
        # priority entry in ``_PATCH_EMBED_CANDIDATES``).
        self.assertEqual(g_dims.hidden_dim_src, st_dims.hidden_dim_src)
        self.assertEqual(g_dims.hidden_dim, st_dims.hidden_dim)
        self.assertEqual(g_dims.in_channels, st_dims.in_channels)
        self.assertEqual(g_dims.patch_size, st_dims.patch_size)

        # And they should match the *torch* convention we wrote.
        self.assertEqual(g_dims.hidden_dim, 1280)
        self.assertEqual(g_dims.in_channels, 4)
        self.assertEqual(g_dims.patch_size, 2)

    def test_arch_detection_match_between_formats(self):
        # Lumina2 detect key tuple includes both ``cap_embedder.1.weight``
        # and ``context_refiner.0.attention.qkv.weight`` — exercise both.
        # ``detect_arch`` takes a set of tensor names and returns
        # ``(arch_name, class_name, invalid)``.
        gguf_tensors, _ = analyze_model.read_model_tensors(self.gguf_path)
        st_tensors, _ = analyze_model.read_model_tensors(self.st_path)
        g_arch = analyze_model.detect_arch({t.name for t in gguf_tensors})
        st_arch = analyze_model.detect_arch({t.name for t in st_tensors})
        self.assertEqual(g_arch, st_arch)
        self.assertEqual(g_arch[1], "ModelLumina2")

    def test_analyze_produces_equal_results(self):
        # End-to-end: identical analyze() output for both inputs.
        g = analyze_model.analyze(self.gguf_path, gpu_vram_gb=8.0,
                                  gpu_name="test", resolutions=[("1024x1024", 1024, 1024)])
        s = analyze_model.analyze(self.st_path, gpu_vram_gb=8.0,
                                  gpu_name="test", resolutions=[("1024x1024", 1024, 1024)])
        self.assertEqual(g.arch, s.arch)
        self.assertEqual(g.arch_class_name, s.arch_class_name)
        self.assertEqual(g.n_tensors, s.n_tensors)
        self.assertEqual(g.n_params, s.n_params)
        self.assertEqual(g.dims.hidden_dim, s.dims.hidden_dim)
        self.assertEqual(g.dims.in_channels, s.dims.in_channels)
        self.assertEqual(g.dims.patch_size, s.dims.patch_size)
        # Weight bytes per quant should be byte-identical because the
        # convert.py F32-promotion rules are shape-driven.
        g_rows = {r.name: r.weight_bytes for r in g.rows}
        s_rows = {r.name: r.weight_bytes for r in s.rows}
        self.assertEqual(g_rows, s_rows)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
