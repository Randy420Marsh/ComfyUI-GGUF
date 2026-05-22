#!/usr/bin/env python3
"""One-shot CLI for the full convert + fix_pad + llama-quantize pipeline.

This is the headless counterpart to ``gguf_gui.py`` and shares the
exact same orchestration code via :mod:`pipeline_lib`, so a CLI run
produces the same output a GUI run would (same Step 0 pre-fold, same
Step 2 skip heuristic, same Step 3 invocation, same Step 4 5-D
re-attach).

Examples
--------
Default Q4_K_M conversion, auto-detect dtype::

    python tools/gguf_pipeline.py \\
        --src model.safetensors \\
        --dst-dir /out

Force a specific quant + bf16 dtype (Ampere+ source already in bf16)::

    python tools/gguf_pipeline.py \\
        --src model.safetensors \\
        --dst-dir /out \\
        --quant Q5_K_M \\
        --dtype bf16

Custom temp dir + keep the intermediates for inspection::

    python tools/gguf_pipeline.py \\
        --src model.safetensors \\
        --dst-dir /out \\
        --temp-dir /scratch \\
        --keep-intermediate
"""

import argparse
import os
import sys

# Allow ``python tools/gguf_pipeline.py`` from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pipeline_lib  # noqa: E402


def _quant_choices_help() -> str:
    """Return a multi-line listing of every supported quant + its description."""
    lines = ["Supported quantization types (passed verbatim to llama-quantize):"]
    width = max(len(name) for name, _ in pipeline_lib.LLAMA_QUANTIZE_TYPES)
    for name, desc in pipeline_lib.LLAMA_QUANTIZE_TYPES:
        lines.append(f"  {name.ljust(width)}  {desc}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gguf_pipeline",
        description=(
            "Headless equivalent of gguf_gui.py: runs convert.py + fix_pad.py "
            "(when needed) + llama-quantize + fix_5d_tensors.py (when needed) "
            "in one command."
        ),
        epilog=_quant_choices_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src", required=True,
        help="Source .safetensors file to convert.",
    )
    parser.add_argument(
        "--dst-dir", required=True,
        help="Directory for the final quantized .gguf output. "
             "Created if missing.",
    )
    parser.add_argument(
        "--temp-dir", default=None,
        help="Directory for intermediate _f16.gguf / _5dfixed.safetensors. "
             "Defaults to <dst-dir>/temp.",
    )
    parser.add_argument(
        "--quant", default=pipeline_lib.DEFAULT_QUANT_TYPE,
        choices=pipeline_lib.QUANTIZE_TYPE_NAMES,
        metavar="TYPE",
        help=(
            f"Output quantization type. Default {pipeline_lib.DEFAULT_QUANT_TYPE}. "
            "Run --help for the full list."
        ),
    )
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "fp16", "bf16"),
        help=(
            "F16 intermediate dtype. 'auto' (default) preserves the source "
            "dtype, 'fp16' / 'bf16' force the choice. Use 'fp16' on Turing "
            "or older NVIDIA (no native bf16)."
        ),
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Do not delete the intermediate _f16.gguf / _5dfixed.safetensors.",
    )
    parser.add_argument(
        "--no-fix-5d", action="store_true",
        help=(
            "Skip the Step 4 fix_5d_tensors.py re-attach pass. The pre-fold "
            "in Step 0 already covers HyVid / Wan, so this is normally a no-op."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    temp_dir = args.temp_dir or os.path.join(args.dst_dir, "temp")
    try:
        final_out = pipeline_lib.run_pipeline(
            src=args.src,
            temp_dir=temp_dir,
            out_dir=args.dst_dir,
            quant_type=args.quant,
            dtype_cli=args.dtype,
            cleanup=not args.keep_intermediate,
            auto_5d_reattach=not args.no_fix_5d,
            log=print,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        return 130
    print(f"\nDone. Output: {final_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
