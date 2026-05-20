# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""
Lightweight GGUF inspector for sanity-checking the output of `convert.py`
(and any GGUF produced by `llama-quantize` further downstream).

Reports:
  * general.architecture       (e.g. 'ernie', 'flux', 'hunyuan')
  * general.file_type          (e.g. MOSTLY_F16, MOSTLY_BF16, MOSTLY_Q4_K_M)
  * tensor count
  * per-tensor-dtype histogram (e.g. F16: 240, F32: 31)

Two scripting-friendly modes:
  --check-no-bf16   exit non-zero if the GGUF still contains any BF16 tensors
                    (useful when you need to guarantee F16-only output for a
                    Turing / RTX 20xx target)
  --verbose         additionally list every tensor with its dtype and shape
"""
import argparse
import os
import sys

import gguf


def _read_str_field(reader, name):
    field = reader.get_field(name)
    if field is None:
        return None
    return str(bytes(field.parts[field.data[-1]]), "utf-8")


def _read_int_field(reader, name):
    field = reader.get_field(name)
    if field is None:
        return None
    arr = field.parts[field.data[-1]]
    if hasattr(arr, "__len__"):
        return int(arr[0])
    return int(arr)


def _ftype_name(value):
    if value is None:
        return None
    try:
        return gguf.LlamaFileType(value).name
    except ValueError:
        return f"UNKNOWN({value})"


def inspect(path, verbose=False):
    reader = gguf.GGUFReader(path)

    arch = _read_str_field(reader, "general.architecture")
    file_type_val = _read_int_field(reader, "general.file_type")
    file_type = _ftype_name(file_type_val)

    histogram = {}
    for tensor in reader.tensors:
        name = tensor.tensor_type.name
        histogram[name] = histogram.get(name, 0) + 1

    print(f"File:         {path}")
    print(f"Architecture: {arch}")
    print(f"File type:    {file_type}")
    print(f"Tensors:      {len(reader.tensors)}")
    print(f"Dtype histogram:")
    for name in sorted(histogram, key=lambda k: (-histogram[k], k)):
        print(f"  {name:<10s}  {histogram[name]}")

    if verbose:
        print("\nTensors (name, dtype, shape):")
        for tensor in reader.tensors:
            shape = tuple(int(d) for d in tensor.shape)
            print(f"  {tensor.name}  {tensor.tensor_type.name}  {shape}")

    return arch, file_type, histogram


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print GGUF architecture, file type, and per-tensor dtype histogram.",
    )
    parser.add_argument("path", help="Path to the .gguf file to inspect.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Additionally list every tensor with its dtype and shape.",
    )
    parser.add_argument(
        "--check-no-bf16", action="store_true",
        help=(
            "Exit non-zero if the GGUF contains any BF16 tensors. Useful as a "
            "guard in convert -> quantize pipelines targeting Turing / RTX 20xx, "
            "which has no native bf16 support."
        ),
    )
    args = parser.parse_args()
    if not os.path.isfile(args.path):
        parser.error(f"file not found: {args.path}")
    return args


def main():
    args = parse_args()
    _arch, _ftype, histogram = inspect(args.path, verbose=args.verbose)

    if args.check_no_bf16 and histogram.get("BF16", 0) > 0:
        print(
            f"\nERROR: {histogram['BF16']} BF16 tensor(s) present "
            f"(expected zero with --check-no-bf16).",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
