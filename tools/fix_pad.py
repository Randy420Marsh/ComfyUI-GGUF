# fix_pad.py - Patch the 1-D x_pad_token / cap_pad_token of Z-Image / Lumina2
# GGUFs so llama-quantize doesn't choke on them.
#
# Run BETWEEN convert.py and llama-quantize for those models, never after.
# Python 3.11 - works on Ubuntu 22.04 / 24.04 and Windows 11.
#
# Versioning notes:
#   - GGUF >= 0.19.0 changed the writer surface; this script handles that.
#   - We deliberately avoid round-tripping the *entire* GGUF through
#     GGUFReader -> GGUFWriter when no padding fix is needed. That round-trip
#     was the root cause of the "basic_ios::clear: iostream error" crash from
#     llama-quantize on Z-Image 0.36 (non-Turbo): re-emitting tensors whose
#     data lives in a numpy view backed by the source mmap occasionally
#     produced a zero-byte entry for a 75 MiB tensor and tripped the
#     quantizer's std::ofstream failbit. When nothing needs fixing we now
#     just copy the file byte-for-byte instead.
import os
import shutil
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter

if len(sys.argv) < 2:
    print("Usage: python fix_pad.py your_f16.gguf")
    sys.exit(1)

path = sys.argv[1]
print(f"Reading {path}...")

reader = GGUFReader(path, "r")
# Insert "_fixed" before the extension only; str.replace(".gguf", ...) would
# also rewrite ".gguf" occurrences earlier in the path (directory names,
# double extensions) and produce an output in the wrong location.
_stem, _ext = os.path.splitext(path)
fixed_path = _stem + "_fixed" + _ext

# First pass: figure out whether any pad tokens are actually 1-D. If neither
# is, the rewrite would be a no-op, and the round-trip itself is risky
# (see header comment), so bail out via a straight file copy instead.
needs_fix = []
for tensor in reader.tensors:
    if tensor.name in ("x_pad_token", "cap_pad_token") and len(tensor.shape) == 1:
        needs_fix.append(tensor.name)

if not needs_fix:
    print(
        "No 1-D pad tokens found -- nothing to fix. "
        "Copying source to _fixed.gguf unchanged so downstream tooling works."
    )
    if os.path.abspath(path) != os.path.abspath(fixed_path):
        shutil.copyfile(path, fixed_path)
    print(f"Done! Saved to {fixed_path}")
    sys.exit(0)

print(f"Will reshape 1-D pad token(s): {', '.join(needs_fix)}")

# Initialize writer with the explicit architecture instead of relying on
# whatever the source put in general.architecture.
arch = "lumina2"
writer = GGUFWriter(fixed_path, arch)

# Copy metadata safely for gguf >= 0.19.0.
for key, field in reader.fields.items():
    if key == "general.architecture":
        continue  # Writer sets this automatically.

    val = field.parts[-1]
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    elif isinstance(val, np.ndarray):
        if len(val) == 1:
            val = val[0]
        else:
            val = val.tolist()

    try:
        writer.add_key_value(key, val)
    except Exception:
        pass  # Safely ignore custom UI keys that GGUF strict-typing rejects.

# Copy tensors + fix pad tokens. For every tensor we materialize the data
# with np.ascontiguousarray() so the writer owns a real copy instead of a
# view into the source mmap -- that detail is what made the previous
# implementation flaky on large files.
for tensor in reader.tensors:
    data = np.ascontiguousarray(tensor.data)
    shape = tensor.shape

    if tensor.name in ("x_pad_token", "cap_pad_token") and len(shape) == 1:
        dim = int(shape[0])
        new_shape = (1, dim)
        print(f"Fixing {tensor.name}: {(dim,)} -> {new_shape}")
        data = data.reshape(new_shape)

    writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"Done! Saved to {fixed_path}")
