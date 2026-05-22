"""On-the-fly analyzer for diffusion checkpoints.

Reads a model's tensor index without loading any weights, detects the
architecture via the same `keys_detect` tuples used by `convert.py`,
extracts model-derived hyperparameters (hidden_dim, depth, patch_size),
and computes a *live* quant -> VRAM matrix against the user's
detected GPU.

Supports two on-disk formats:
  - `.safetensors` -- parses the JSON header at offset 8.
  - `.gguf` (any architecture) -- uses `gguf.GGUFReader` to walk the
    tensor index. Useful when the user wants to re-evaluate a quant
    decision against an already-converted intermediate or final GGUF
    (e.g. compare the city96 pre-quantized Z-Image Turbo to a fresh
    Z-Image 0.36 F16 conversion).

Nothing in this module is hardcoded per architecture or per model: the
weight cost is summed over the actual tensors in the file under
convert.py's own quantize-vs-preserve rules, and the activation cost is
derived from the model's own hidden_dim / patch_size / target
resolution.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

# Reuse the canonical architecture taxonomy from convert.py so detection
# stays in lock-step with the conversion code. Importing the file by
# path keeps this module loadable from contexts where `tools` is not a
# package (e.g. the GUI runs convert.py via subprocess, not import).
import importlib.util as _ilu
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_convert", os.path.join(_HERE, "convert.py"))
_convert = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_convert)
ARCH_LIST = _convert.arch_list
QUANTIZATION_THRESHOLD = _convert.QUANTIZATION_THRESHOLD  # mirrors convert.py
MAX_TENSOR_DIMS = _convert.MAX_TENSOR_DIMS


# ── Bits-per-weight table ────────────────────────────────────────────────────
#
# Sourced from ggml's published quant definitions (block sizes and per-block
# byte counts in ggml-quants.c). For mixed K-quants the value is the
# aggregate average reported upstream.
#
# Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 are ARM-only repacks of Q4_0; the on-disk
# size is identical so they share the same bpw.
BITS_PER_WEIGHT = {
    "F32":      32.0,
    "F16":      16.0,
    "BF16":     16.0,
    "Q8_0":     8.5,
    "Q6_K":     6.5625,
    "Q5_K":     5.5,
    "Q5_K_M":   5.5,
    "Q5_K_S":   5.5,
    "Q5_1":     6.0,
    "Q5_0":     5.5,
    "Q4_K":     4.85,
    "Q4_K_M":   4.85,
    "Q4_K_S":   4.5,
    "Q4_1":     5.0,
    "Q4_0":     4.5,
    "Q4_0_4_4": 4.5,
    "Q4_0_4_8": 4.5,
    "Q4_0_8_8": 4.5,
    "Q3_K":     3.91,
    "Q3_K_L":   4.25,
    "Q3_K_M":   3.91,
    "Q3_K_S":   3.44,
    "Q2_K":     2.625,
    "Q2_K_S":   2.625,
    "IQ4_NL":   4.5,
    "IQ4_XS":   4.25,
    "IQ3_M":    3.66,
    "IQ3_S":    3.44,
    "IQ3_XS":   3.3,
    "IQ3_XXS":  3.06,
    "IQ2_M":    2.7,
    "IQ2_S":    2.5,
    "IQ2_XS":   2.31,
    "IQ2_XXS":  2.06,
    "IQ1_M":    1.75,
    "IQ1_S":    1.56,
    "TQ2_0":    2.06,
    "TQ1_0":    1.69,
    # COPY = pass-through; assume the convert.py intermediate (F16) is what
    # the user would copy unchanged.
    "COPY":     16.0,
}


# Bytes per element for the safetensors source dtypes we may encounter.
# Used only for sanity / reporting; actual conversion goes through
# convert.py which collapses everything to F16/BF16/F32 first.
SAFETENSORS_BYTES = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "I8": 1, "U8": 1, "BOOL": 1,
}


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    n_params: int


@dataclass
class ModelDims:
    """Hyperparameters derived directly from the safetensors header.

    `None` fields could not be reliably inferred and the activation
    estimator should fall back to a conservative default.
    """
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    patch_size: Optional[int] = None
    in_channels: Optional[int] = None
    # Key the dims were derived from, for transparency in the UI.
    hidden_dim_src: Optional[str] = None
    patch_size_src: Optional[str] = None


@dataclass
class QuantRow:
    name: str
    weight_bytes: int
    activations_bytes: dict[str, int]   # resolution_label -> bytes
    total_bytes: dict[str, int]          # resolution_label -> bytes (weight + act + overhead)
    fits: dict[str, bool]                # resolution_label -> bool (within VRAM - headroom)


@dataclass
class AnalysisResult:
    path: str
    file_size: int
    n_tensors: int
    n_params: int
    dtype_histogram: dict[str, int]      # dtype -> param count
    arch: Optional[str]
    arch_class_name: Optional[str]
    invalid: bool                        # banned-keys hit during arch detect
    dims: ModelDims
    rows: list[QuantRow]
    resolutions: list[tuple[str, int, int]]   # (label, w, h)
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    headroom_gb: float
    comfyui_overhead_mb: int
    recommended: Optional[str]           # quant name
    activation_formula: str              # human-readable formula
    notes: list[str] = field(default_factory=list)


# ── safetensors header reader ───────────────────────────────────────────────


GGUF_MAGIC = b"GGUF"


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, file_size). Reads only the 8-byte length + JSON
    header; does NOT touch tensor data."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("safetensors file too short to contain a header length")
        (hdr_len,) = struct.unpack("<Q", raw)
        if hdr_len > 100 * 1024 * 1024:
            raise ValueError(f"safetensors header length implausible: {hdr_len}")
        hdr = f.read(hdr_len)
        if len(hdr) != hdr_len:
            raise ValueError("safetensors header truncated")
    return json.loads(hdr.decode("utf-8")), file_size


def _detect_format(path: str) -> str:
    """Return ``"gguf"`` or ``"safetensors"`` for ``path``.

    Prefers the 4-byte magic over the extension so a misnamed file still
    routes correctly. Raises ValueError when neither matches.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == GGUF_MAGIC:
        return "gguf"
    # safetensors has no magic; fall back to the extension and let the
    # downstream parser raise a precise error if that's wrong too.
    if path.lower().endswith(".gguf"):
        return "gguf"
    if path.lower().endswith(".safetensors"):
        return "safetensors"
    # Last-chance guess: try parsing the safetensors header length.
    return "safetensors"


def read_gguf_tensors(path: str) -> tuple[list["TensorInfo"], int]:
    """Return (tensors, file_size) by walking a GGUF tensor index.

    Uses ``gguf.GGUFReader`` (already a hard dep of this repo) so the
    parser stays in lock-step with the format version that ``convert.py``
    and ``fix_pad.py`` emit. Imports are local to keep the safetensors-
    only path free of the gguf import cost.
    """
    import gguf as _gguf  # local import: hot path for the safetensors case
    file_size = os.path.getsize(path)
    reader = _gguf.GGUFReader(path, "r")
    out: list[TensorInfo] = []
    for t in reader.tensors:
        shape = tuple(int(d) for d in t.shape)
        nelem = 1
        for d in shape:
            nelem *= d
        # ``t.tensor_type`` is a ``GGMLQuantizationType`` enum; ``.name``
        # already matches our BITS_PER_WEIGHT keys ("F16", "BF16", "Q4_K",
        # "F32", …) for everything we need to render the histogram.
        dtype_name = t.tensor_type.name
        out.append(TensorInfo(
            name=t.name,
            dtype=dtype_name,
            shape=shape,
            n_params=nelem,
        ))
    return out, file_size


def read_model_tensors(path: str) -> tuple[list["TensorInfo"], int]:
    """Format-agnostic helper. Returns ``(tensors, file_size)``.

    Routes to ``read_safetensors_header`` + ``tensors_from_header`` or
    to ``read_gguf_tensors`` based on the file magic / extension.
    """
    fmt = _detect_format(path)
    if fmt == "gguf":
        return read_gguf_tensors(path)
    hdr, file_size = read_safetensors_header(path)
    return tensors_from_header(hdr), file_size


def tensors_from_header(hdr: dict) -> list[TensorInfo]:
    out = []
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        shape = tuple(info.get("shape", []))
        n = 1
        for d in shape:
            n *= d
        out.append(TensorInfo(
            name=name,
            dtype=info.get("dtype", "F16"),
            shape=shape,
            n_params=n,
        ))
    return out


# ── architecture + dim detection ────────────────────────────────────────────


def detect_arch(keys: set[str]) -> tuple[Optional[str], Optional[str], bool]:
    """Return (arch_name, class_name, invalid).

    Mirrors convert.py:is_model_arch — the same ordering of `arch_list` is
    used so detection here can never disagree with the converter.
    """
    for cls in ARCH_LIST:
        for match_list in cls.keys_detect:
            if all(k in keys for k in match_list):
                invalid = any(k in keys for k in cls.keys_banned)
                return cls.arch, cls.__name__, invalid
    return None, None, False


# Candidate key suffixes that typically hold the patch-embedding conv (4D
# weight: (hidden, channels, patch, patch)) or a Linear that operates on
# pre-patched input (2D weight: (hidden, channels * patch^2)). Order
# matters: more specific names first.
_PATCH_EMBED_CANDIDATES = (
    "x_embedder.proj.weight",
    "x_embedder.weight",
    "patch_embed.proj.weight",
    "patch_embedder.proj.weight",
    "img_in.proj.weight",
    "img_in.weight",
    "input_blocks.0.0.weight",   # SD1/SDXL UNet first conv
)


def extract_dims(tensors: list[TensorInfo]) -> ModelDims:
    by_name = {t.name: t for t in tensors}
    dims = ModelDims()

    # hidden_dim + patch_size from a patch-embedder key.
    for suffix in _PATCH_EMBED_CANDIDATES:
        match = next((t for t in tensors if t.name.endswith(suffix)), None)
        if match is None:
            continue
        s = match.shape
        if len(s) == 4:
            # (out_channels, in_channels, kh, kw) — kh == kw == patch_size.
            dims.hidden_dim = s[0]
            dims.in_channels = s[1]
            dims.patch_size = s[2]
            dims.hidden_dim_src = match.name
            dims.patch_size_src = match.name
            break
        if len(s) == 2:
            # Linear over pre-patched input: out = hidden, in = ch * p^2.
            dims.hidden_dim = s[0]
            dims.hidden_dim_src = match.name
            # patch_size not directly recoverable — leave as None;
            # the activation estimator will default to 2 (the DiT standard).
            break

    # Fallback: hidden_dim from any to_q.weight (most attention impls).
    if dims.hidden_dim is None:
        for t in tensors:
            if t.name.endswith("to_q.weight") and len(t.shape) == 2:
                dims.hidden_dim = t.shape[0]
                dims.hidden_dim_src = t.name
                break

    # Layer count: max index N seen after `layers.` / `blocks.` /
    # `transformer_blocks.` / `double_blocks.` / `single_blocks.` etc.
    max_idx = -1
    layer_prefixes = (
        "layers.", "blocks.", "transformer_blocks.", "single_blocks.",
        "double_blocks.", "joint_transformer_blocks.", "input_blocks.",
    )
    for name in by_name:
        for pref in layer_prefixes:
            i = name.find(pref)
            if i < 0:
                continue
            rest = name[i + len(pref):]
            digits = rest.split(".", 1)[0]
            if digits.isdigit():
                max_idx = max(max_idx, int(digits))
    if max_idx >= 0:
        dims.num_layers = max_idx + 1

    return dims


# ── weight cost ─────────────────────────────────────────────────────────────


def _tensor_keeps_f32(t: TensorInfo, hiprec_keys: list[str]) -> bool:
    """Returns True if convert.py would force this tensor to F32 regardless
    of the user's chosen quant. Mirrors convert.py:handle_tensors."""
    if len(t.shape) == 1:
        return True
    if t.n_params <= QUANTIZATION_THRESHOLD:
        return True
    if any(h in t.name for h in hiprec_keys):
        return True
    return False


def weight_cost_bytes(tensors: list[TensorInfo], quant_type: str,
                      arch_class_name: Optional[str]) -> int:
    """Sum bytes over all tensors under the given quant choice.

    Applies convert.py's quantize-vs-preserve rules verbatim:
      - 1D tensors -> F32
      - tiny tensors (<= 1024 elements) -> F32
      - keys_hiprec match for the detected arch -> F32
      - everything else -> chosen quant's bpw
    """
    bpw = BITS_PER_WEIGHT.get(quant_type, 16.0)
    f32_bpw = 32.0

    hiprec = []
    if arch_class_name:
        cls = next((c for c in ARCH_LIST if c.__name__ == arch_class_name), None)
        if cls is not None:
            hiprec = list(cls.keys_hiprec)

    total_bits = 0
    for t in tensors:
        if len(t.shape) > MAX_TENSOR_DIMS:
            # 5D+ tensors get folded by fix_5d_tensors.py; the post-fold
            # tensor has the same param count so this is still a fine
            # estimate.
            pass
        if _tensor_keeps_f32(t, hiprec):
            total_bits += t.n_params * f32_bpw
        else:
            total_bits += t.n_params * bpw
    return total_bits // 8


# ── activation cost ─────────────────────────────────────────────────────────
#
# Single transparent formula, model-derived. Documented in the UI tooltip:
#
#   seq_len = (H / patch) * (W / patch)
#   peak_act = seq_len * hidden_dim * 3 * 2 bytes
#
# Rationale: with PyTorch SDPA / xformers (ComfyUI's default on Turing+),
# attention scores are computed chunk-by-chunk and never materialise, so
# the peak is dominated by the residual stream + held activations during
# a single layer's forward. Empirically that's ~3x the residual stream
# at fp16. Layer count does NOT scale linearly because ComfyUI runs
# layers sequentially and frees intermediates eagerly. The 3x factor is
# a known-conservative upper bound; real peak on a 2070S with --lowvram
# is often ~50% lower.
#
# Caveats surfaced in the UI:
#   - Estimate is ±25%, lower with --lowvram, higher without.
#   - Assumes SDPA / flash attention. Naive O(seq^2) attention would
#     blow up — not the default in ComfyUI.
#   - Does NOT include text encoder or VAE (assumed loaded separately as
#     GGUFs and not co-resident during the diffusion step).

ACTIVATION_BYTES_PER_TOKEN_HIDDEN = 6   # 3 * 2 (fp16)

DEFAULT_PATCH_SIZE = 2
DEFAULT_HIDDEN_FALLBACK = 2048
DEFAULT_VAE_SCALE = 8           # every arch in IMG_ARCH_LIST uses an 8x VAE
COMFY_OVERHEAD_MB_DEFAULT = 400


def activation_bytes(hidden_dim: Optional[int], patch_size: Optional[int],
                     width: int, height: int,
                     vae_scale: int = DEFAULT_VAE_SCALE) -> int:
    """Estimate peak activation bytes during one diffusion-step forward.

    `width` / `height` are the user-facing PIXEL resolution. The diffusion
    model itself operates on the VAE-encoded latent (1/8 of the pixel
    resolution for every arch in IMG_ARCH_LIST), then patchifies that with
    `patch_size`. So the token count seen by the transformer at
    1024x1024 px is (1024/8/2)^2 = 4096, not (1024/2)^2 = 262144.

    Without the VAE step this number would be ~64x too large and the
    recommendation logic would push every realistic checkpoint down to
    the smallest quant.
    """
    p = patch_size or DEFAULT_PATCH_SIZE
    h = hidden_dim or DEFAULT_HIDDEN_FALLBACK
    latent_w = max(width // vae_scale, 1)
    latent_h = max(height // vae_scale, 1)
    seq = (latent_w // p) * (latent_h // p)
    return seq * h * ACTIVATION_BYTES_PER_TOKEN_HIDDEN


# ── top-level analysis ──────────────────────────────────────────────────────


def analyze(path: str,
            gpu_vram_gb: Optional[float] = None,
            gpu_name: Optional[str] = None,
            resolutions: Optional[list[tuple[str, int, int]]] = None,
            headroom_gb: float = 1.0,
            comfyui_overhead_mb: int = COMFY_OVERHEAD_MB_DEFAULT,
            quant_types: Optional[list[str]] = None) -> AnalysisResult:
    """Run the full analysis pass on a safetensors checkpoint.

    `gpu_vram_gb=None` means no fit/recommendation column will be filled.
    `resolutions` defaults to [(1024², 1024, 1024), (1536x1024, 1536, 1024)].
    `quant_types` defaults to a curated short list spanning the practical
    range; pass any subset of BITS_PER_WEIGHT.keys() to override.
    """
    if resolutions is None:
        resolutions = [
            ("1024x1024", 1024, 1024),
            ("1536x1024", 1536, 1024),
        ]
    if quant_types is None:
        quant_types = [
            "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S",
            "IQ3_M", "Q3_K_M", "Q3_K_S", "IQ2_M", "Q2_K",
        ]
    # Sort defensively by descending bpw so the "first fitting row wins"
    # recommendation loop never skips a higher-quality option just because
    # the caller passed the list out of order.
    quant_types = sorted(
        quant_types,
        key=lambda q: BITS_PER_WEIGHT.get(q, float("inf")),
        reverse=True,
    )

    tensors, file_size = read_model_tensors(path)
    keys = {t.name for t in tensors}

    arch, klass, invalid = detect_arch(keys)
    dims = extract_dims(tensors)

    n_params = sum(t.n_params for t in tensors)
    dtype_hist: dict[str, int] = {}
    for t in tensors:
        dtype_hist[t.dtype] = dtype_hist.get(t.dtype, 0) + t.n_params

    notes: list[str] = []
    if arch is None:
        notes.append("Architecture not recognised — VRAM estimates still apply, "
                     "but keys_hiprec promotions can't be modelled. The actual "
                     "GGUF size may be 1-3% larger than shown.")
    if invalid:
        notes.append(f"This checkpoint matches {arch!r} but also contains a "
                     "banned key (reference-implementation variant). convert.py "
                     "will refuse to convert it as-is.")
    if dims.hidden_dim is None:
        notes.append("Could not derive hidden_dim from any known key — "
                     f"falling back to {DEFAULT_HIDDEN_FALLBACK} for the "
                     "activation estimate (heuristic).")
    if dims.patch_size is None:
        notes.append(f"Could not derive patch_size — assuming {DEFAULT_PATCH_SIZE} "
                     "(DiT standard). If the model uses a different patch size "
                     "the activation budget will be off by patch_ratio².")

    overhead_bytes = comfyui_overhead_mb * 1024 * 1024
    headroom_bytes = int(headroom_gb * 1024 * 1024 * 1024)

    rows: list[QuantRow] = []
    for q in quant_types:
        wb = weight_cost_bytes(tensors, q, klass)
        acts: dict[str, int] = {}
        totals: dict[str, int] = {}
        fits: dict[str, bool] = {}
        for label, w, h in resolutions:
            ab = activation_bytes(dims.hidden_dim, dims.patch_size, w, h)
            acts[label] = ab
            tot = wb + ab + overhead_bytes
            totals[label] = tot
            if gpu_vram_gb is not None:
                budget = int(gpu_vram_gb * 1024 * 1024 * 1024) - headroom_bytes
                fits[label] = tot <= budget
            else:
                fits[label] = False
        rows.append(QuantRow(name=q, weight_bytes=wb,
                             activations_bytes=acts,
                             total_bytes=totals,
                             fits=fits))

    # Recommendation: highest-quality (= highest bpw) quant whose largest
    # configured resolution still fits within (VRAM - headroom). The
    # `quant_types` input order already runs heavy -> light, so the first
    # fitting row wins.
    recommended: Optional[str] = None
    if gpu_vram_gb is not None:
        worst_label = resolutions[-1][0]
        for row in rows:
            if row.fits.get(worst_label):
                recommended = row.name
                break

    formula = (
        "weight = sum_tensors(n_params * bpw[quant] / 8, "
        "F32 if 1D / <=1024 elems / keys_hiprec)\n"
        "latent_seq = (W/8/patch) * (H/8/patch)        "
        "(8x VAE downsample applied before patchification)\n"
        "activations = latent_seq * hidden_dim * 3 * 2 bytes   "
        "(SDPA / flash attention; ±25% on Turing with --lowvram)\n"
        "total = weight + activations + 400 MB ComfyUI overhead\n"
        "fits = total <= (VRAM - 1 GB headroom)"
    )

    return AnalysisResult(
        path=path,
        file_size=file_size,
        n_tensors=len(tensors),
        n_params=n_params,
        dtype_histogram=dtype_hist,
        arch=arch,
        arch_class_name=klass,
        invalid=invalid,
        dims=dims,
        rows=rows,
        resolutions=resolutions,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        headroom_gb=headroom_gb,
        comfyui_overhead_mb=comfyui_overhead_mb,
        recommended=recommended,
        activation_formula=formula,
        notes=notes,
    )


# Pretty-printing helpers for CLI / GUI consumers.


def fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    return str(n)


if __name__ == "__main__":
    # CLI: python tools/analyze_model.py <path> [vram_gb]
    import sys
    if len(sys.argv) < 2:
        print("Usage: analyze_model.py <path.safetensors> [vram_gb]")
        sys.exit(2)
    path = sys.argv[1]
    vram = float(sys.argv[2]) if len(sys.argv) > 2 else None
    r = analyze(path, gpu_vram_gb=vram)
    print(f"File:    {r.path}  ({fmt_bytes(r.file_size)})")
    print(f"Tensors: {r.n_tensors}    Params: {fmt_params(r.n_params)}")
    print(f"Arch:    {r.arch}  ({r.arch_class_name}){'  INVALID' if r.invalid else ''}")
    print(f"hidden_dim={r.dims.hidden_dim}  num_layers={r.dims.num_layers}"
          f"  patch_size={r.dims.patch_size}  in_channels={r.dims.in_channels}")
    print(f"hidden_dim source key: {r.dims.hidden_dim_src}")
    print(f"dtype histogram (params): {r.dtype_histogram}")
    if vram is not None:
        print(f"GPU: {r.gpu_name}  VRAM={vram} GB  headroom={r.headroom_gb} GB  "
              f"overhead={r.comfyui_overhead_mb} MB")
        print(f"Recommended: {r.recommended}")
    print()
    print(f"{'quant':<10}{'weight':>12}", end="")
    for label, _, _ in r.resolutions:
        print(f"  {'act@'+label:>12}{'tot@'+label:>12}{'fit':>5}", end="")
    print()
    for row in r.rows:
        print(f"{row.name:<10}{fmt_bytes(row.weight_bytes):>12}", end="")
        for label, _, _ in r.resolutions:
            print(f"  {fmt_bytes(row.activations_bytes[label]):>12}"
                  f"  {fmt_bytes(row.total_bytes[label]):>12}"
                  f"  {('Y' if row.fits[label] else 'n') if vram else '-':>5}", end="")
        print()
    print()
    print("Formula:")
    print(r.activation_formula)
    for n in r.notes:
        print(f"  note: {n}")
