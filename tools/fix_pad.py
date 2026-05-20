# fix_pad.py - Fixes x_pad_token & cap_pad_token shape for Z-Image models
# Python 3.11 • Works on Ubuntu 22.04-24.04 & Win11
# Run BEFORE quantizing
# fix_pad.py - FIXED FOR GGUF 0.19.0
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter

if len(sys.argv) < 2:
    print("Usage: python fix_pad.py your_f16.gguf")
    sys.exit(1)

path = sys.argv[1]
print(f"Reading {path}...")

reader = GGUFReader(path, "r")
fixed_path = path.replace(".gguf", "_fixed.gguf")

# Initialize writer with the explicit architecture instead of missing gguf_version
arch = "lumina2"
writer = GGUFWriter(fixed_path, arch)

# Copy metadata safely for gguf >= 0.19.0
for key, field in reader.fields.items():
    if key == "general.architecture":
        continue # Writer sets this automatically
    
    # Extract data securely depending on the field type
    val = field.parts[-1]
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    elif isinstance(val, np.ndarray):
        if len(val) == 1:
            val = val[0]
        else:
            val = val.tolist()
            
    try:
        writer.add_key_value(key, val)
    except Exception as e:
        pass # Safely ignore custom UI keys that GGUF strict-typing rejects

# Copy tensors + fix pad tokens
for tensor in reader.tensors:
    data = tensor.data
    shape = tensor.shape

    if tensor.name in ("x_pad_token", "cap_pad_token") and len(shape) == 1:
        dim = shape[0]
        new_shape = (1, dim)
        print(f"Fixing {tensor.name}: {shape} -> {new_shape}")
        data = data.reshape(new_shape)

    writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"Done! Saved to {fixed_path}")
