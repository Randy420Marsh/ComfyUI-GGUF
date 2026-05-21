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
  --metadata        dump every KV metadata field (architecture name, head
                    counts, context length, tokenizer model, etc). Use this
                    instead of safetensors-based readers, which won't open
                    GGUFs (the two formats are unrelated).
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


def _format_field_value(field):
    """Render a gguf.ReaderField as a short string for the metadata dump.

    Falls back gracefully on arrays / unknown types so we never blow up
    on giant tokenizer-vocab fields.
    """
    try:
        t0 = field.types[0]
        if t0 == gguf.GGUFValueType.STRING:
            return repr(str(bytes(field.parts[field.data[-1]]), "utf-8"))
        if t0 == gguf.GGUFValueType.ARRAY:
            inner = field.types[1] if len(field.types) > 1 else None
            inner_name = getattr(inner, "name", str(inner))
            return f"<ARRAY[{inner_name}] len={len(field.data)}>"
        if t0 == gguf.GGUFValueType.BOOL:
            return repr(bool(field.parts[field.data[-1]][0]))
        arr = field.parts[field.data[-1]]
        v = arr[0] if hasattr(arr, "__len__") else arr
        if t0 in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            return repr(float(v))
        return repr(int(v))
    except Exception as e:
        return f"<unreadable: {e}>"


def inspect(path, verbose=False, metadata=False):
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

    if metadata:
        print("\nMetadata (KV fields):")
        for field_name in sorted(reader.fields):
            field = reader.get_field(field_name)
            if field is None:
                continue
            print(f"  {field_name} = {_format_field_value(field)}")

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
        "--metadata", action="store_true",
        help=(
            "Dump every KV metadata field (architecture, head counts, "
            "context length, tokenizer model, etc). Replaces the need "
            "for an external safetensors-based reader -- those won't "
            "open GGUFs anyway."
        ),
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
    _arch, _ftype, histogram = inspect(
        args.path, verbose=args.verbose, metadata=args.metadata,
    )

    if args.check_no_bf16 and histogram.get("BF16", 0) > 0:
        print(
            f"\nERROR: {histogram['BF16']} BF16 tensor(s) present "
            f"(expected zero with --check-no-bf16).",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
