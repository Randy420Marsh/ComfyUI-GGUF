#!/usr/bin/env python3
"""Extract the LTX-2 / LTX-2.3 `text_embedding_projection` tensors from a
Lightricks checkpoint into a standalone safetensors file.

Why: the LTX text encoder is Gemma-3-12B *plus* a `text_embedding_projection`
layer that maps the stacked hidden states of all 49 Gemma layers (4D:
[batch, 49, seq, 3840]) down to the 3D embedding the LTXAV connectors expect.
That projection is not a Gemma tensor, so llama.cpp GGUF conversion drops it,
and Lightricks/Comfy-Org only distribute it embedded in the full checkpoint
(top-level `text_embedding_projection.*` keys next to `model.diffusion_model.*`).

A Gemma GGUF loaded alone therefore crashes at sampling time with
`RuntimeError: Tensors must have same number of dimensions: got 4 and 3`.
The fix is `DualCLIPLoader (GGUF)` with type `ltxv`:
  clip_name1 = the Gemma-3-12B GGUF   (must be first: tokenizer sidecar)
  clip_name2 = the file this script produces

Usage:
  python tools/extract_ltx_projection.py \
      --src /models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors \
      --dst /models/text_encoders/ltx-2.3-text_embedding_projection.safetensors

Reads only the needed tensors (~2.3 GB for LTX 2.3 dual_linear), not the
whole checkpoint. Any checkpoint of the same model family works — dev-fp8
and distilled-1.1 carry identical projection weights.
"""

import argparse
import os

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "text_embedding_projection."


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="LTX checkpoint .safetensors")
    parser.add_argument("--dst", default=None,
                        help="output .safetensors (default: <src-dir>/<src-stem>-text_embedding_projection.safetensors)")
    args = parser.parse_args()

    dst = args.dst
    if dst is None:
        stem = os.path.splitext(os.path.basename(args.src))[0]
        dst = os.path.join(os.path.dirname(args.src), f"{stem}-text_embedding_projection.safetensors")

    out = {}
    with safe_open(args.src, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                out[k] = f.get_tensor(k)

    if not out:
        raise SystemExit(
            f"No '{PREFIX}*' tensors found in {args.src} — this is not an LTX-2/2.3 "
            f"checkpoint (the projection only ships inside the full Lightricks "
            f"checkpoints, not in the Comfy-Org gemma text-encoder files)."
        )

    for k, v in sorted(out.items()):
        print(f"  {k}  {v.dtype}  {tuple(v.shape)}")
    save_file(out, dst)
    print(f"written: {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
