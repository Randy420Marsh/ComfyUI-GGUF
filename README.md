# ComfyUI-GGUF (Randy420Marsh fork)
GGUF Quantization support for native ComfyUI models

This is a fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) with additional architecture support and conversion tooling that is not present upstream. Highlights specific to this fork:

- Text-encoder GGUF support for `mistral3` (Ministral-3-3B, used by ERNIE-Image), plus `Mistral`-family `tekken` tokenizer reconstruction from GGUF metadata.
- Gemma-4 text-encoder loading via `tokenizer.json` sidecar (fixes the `'str' object has no attribute 'decode'` crash you get with the upstream loader).
- Optional `mmproj_name` picker on `CLIPLoader (GGUF)` for explicit multimodal-projector selection when filename auto-discovery cannot match it.
- `tools/convert.py` extended to support ERNIE-Image and ComfyUI scaled-fp8 dequantization.
- `tools/gguf_gui.py` — a Qt GUI front-end for `convert.py` with bf16 auto-detection, a full `llama-quantize` output-type selector, and an **Analyze** button that picks a quant target from the model's metadata.
- `tools/inspect_gguf.py --metadata` for inspecting per-architecture GGUF metadata (replaces a safetensors-only script that does not work on GGUF).
- `docs/CONVERSION_GUIDE.md` — long-form per-model walkthrough.

Use this repo's URL when installing, cloning the tools, or filing bug reports about the fork-specific features. Issues that exist in the upstream loader (everything outside the bullets above) should be reported to [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/issues) where the original author can triage them.

## Documentation

The repo carries three layers of documentation depending on what you're trying to do:

| Where | Audience | Content |
|---|---|---|
| **This file** (`README.md`) | First-time visitors | What the fork is, what's new vs. upstream, how to install the custom node, list of pre-quantized model links. |
| [`tools/README.md`](https://github.com/Randy420Marsh/ComfyUI-GGUF/blob/main/tools/README.md) | Anyone converting their own models | Setup reference doc for `convert.py` + `gguf_gui.py` + the patched `llama-quantize` build. Covers venv, dependencies, `LD_LIBRARY_PATH`, GUI controls, Analyze button, CLI invocations, troubleshooting, platform notes (VS2022, CUDA, macOS). |
| [`docs/CONVERSION_GUIDE.md`](https://github.com/Randy420Marsh/ComfyUI-GGUF/blob/main/docs/CONVERSION_GUIDE.md) | Anyone converting a specific model | Long-form per-model walkthrough: Flux, SD3 / SD3.5, ERNIE-Image (Ministral-3-3B text encoder), Z-Image / Lumina2 / RedCraft ZiB, Hunyuan Video, Wan 2.1, plus the math behind the Analyze recommendation and a quant-types reference. |
| **Wiki** ([landing page](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki)) | Anyone who prefers a browseable nav | Mirror of the above plus the shortcut build recipe. |

### Wiki pages

- **[Home](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki)** — index + fork-features → PR table mapping each addition in this fork to the PR that introduced it.
- **[Build the patched llama-quantize](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki/Build-llama-quantize)** — build recipe using the **pre-patched** [`city96` branch](https://github.com/Randy420Marsh/llama.cpp/tree/city96) of [`Randy420Marsh/llama.cpp`](https://github.com/Randy420Marsh/llama.cpp) so you don't have to `git clone llama.cpp + git checkout tags/b3962 + git apply lcpp.patch` by hand. Covers CPU / CUDA / Windows builds, `LD_LIBRARY_PATH` setup, and a smoke test. The manual `lcpp.patch` route is still documented inside [`tools/README.md`](https://github.com/Randy420Marsh/ComfyUI-GGUF/blob/main/tools/README.md#3-build-the-patched-llama-quantize) as a fallback.
- **[Conversion-Guide](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki/Conversion-Guide)** — wiki mirror of [`docs/CONVERSION_GUIDE.md`](https://github.com/Randy420Marsh/ComfyUI-GGUF/blob/main/docs/CONVERSION_GUIDE.md) with the relative `../tools/README.md` links rewritten to absolute URLs so they resolve from the wiki.

The in-repo `tools/README.md` and `docs/CONVERSION_GUIDE.md` are the source of truth; the wiki pages mirror them for browseability. If you're filing a bug or PR, edit the in-repo files.

---

These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/70d16d97-c522-4ef4-9435-633f128644c8)

Note: The "Force/Set CLIP Device" is **NOT** part of this node pack. Do not install it if you only have one GPU. Do not set it to cuda:0 then complain about OOM errors if you do not undestand what it is for. There is not need to copy the workflow above, just use your own workflow and replace the stock "Load Diffusion Model" with the "Unet Loader (GGUF)" node.

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/Randy420Marsh/ComfyUI-GGUF
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/Randy420Marsh/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

On MacOS sequoia, torch 2.4.1 seems to be required, as 2.6.X nightly versions cause a "M1 buffer is not large enough" error. See [this upstream issue](https://github.com/city96/ComfyUI-GGUF/issues/107) for more information/workarounds.

## Usage

Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.

LoRA loading is experimental but it should work with just the built-in LoRA loader node(s).

Pre-quantized models:

- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)

Initial support for quantizing T5 has also been added recently, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.

- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)

See the instructions in the [tools](https://github.com/Randy420Marsh/ComfyUI-GGUF/tree/main/tools) folder for how to create your own quants. Long-form per-model walkthroughs (Flux, SD3.5, ERNIE-Image, Lumina/Gemma, Wan/Hunyuan-Video, etc.) live in [`docs/CONVERSION_GUIDE.md`](https://github.com/Randy420Marsh/ComfyUI-GGUF/blob/main/docs/CONVERSION_GUIDE.md), or on the [wiki](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki) which also has a dedicated [Build the patched llama-quantize](https://github.com/Randy420Marsh/ComfyUI-GGUF/wiki/Build-llama-quantize) page that uses the pre-patched [`city96` branch](https://github.com/Randy420Marsh/llama.cpp/tree/city96) of [`Randy420Marsh/llama.cpp`](https://github.com/Randy420Marsh/llama.cpp) — no manual `git apply` step required.
