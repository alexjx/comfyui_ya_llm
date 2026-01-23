# LlamaCpp Node - Quick Start Guide

Get started with local LLM inference using llama.cpp in ComfyUI.

## 1. Install Dependencies

### Install llama-cpp-python with CUDA

```bash
# For CUDA 12.1 (most common)
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

See [LLAMACPP_INSTALL.md](LLAMACPP_INSTALL.md) for other CUDA versions and installation options.

### Verify Installation

```bash
python -c "from llama_cpp import Llama; print('Success!')"
```

## 2. Download Models

### Where to Put Models

Place GGUF model files in:
```
ComfyUI/models/LLM/
```

The node will automatically detect all `.gguf` files in this directory.

### Recommended Models

**Text Generation (7B-8B):**
- [Qwen2.5-7B-Instruct](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF) - Excellent general purpose
- [Llama-3.1-8B-Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) - Strong reasoning
- [Mistral-7B-Instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) - Fast and efficient

**Download Example:**
```bash
cd ComfyUI/models/LLM/

# Download Qwen2.5-7B-Instruct Q4_K_M (recommended quantization)
wget https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

### Vision Models (Optional)

For image understanding, download a vision model + mmproj file:

```bash
cd ComfyUI/models/LLM/

# Llava 1.6 Mistral 7B
wget https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-1.6-mistral-7b.Q4_K_M.gguf
wget https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf
```

## 3. Use in ComfyUI

### Basic Text Generation

1. Start/restart ComfyUI
2. Add node: `Yet Another LLM > LlamaCpp Generate`
3. Configure:
   - **model**: Select your .gguf file from dropdown
   - **prompt**: Enter your question/instruction
   - **n_gpu_layers**: Set to `-1` for full GPU acceleration
   - **unload_model**: Keep `True` to free memory after generation
4. Run!

### Example Workflow

```
[TextTemplate] → prompt → [LlamaCppGenerate] → response → [Display Text]
                           ↓ model: Qwen2.5-7B-Q4_K_M
                           ↓ n_gpu_layers: -1
                           ↓ temperature: 1.0
```

### Vision Model Usage

1. Add node: `LlamaCpp Generate`
2. Configure:
   - **model**: Select vision model (e.g., llava-1.6-mistral-7b.Q4_K_M.gguf)
   - **clip_model**: Select mmproj file (e.g., mmproj-model-f16.gguf)
   - **images**: Connect image input
   - **prompt**: "Describe this image in detail"
3. Run!

### Example Vision Workflow

```
[Load Image] → images → [LlamaCppGenerate] → response → [Display Text]
                         ↓ model: llava-1.6-mistral-7b.Q4_K_M.gguf
                         ↓ clip_model: mmproj-model-f16.gguf
                         ↓ prompt: "What do you see in this image?"
```

## 4. Parameters Explained

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | - | Your instruction/question for the model |
| `model` | - | Select GGUF model from dropdown |
| `n_gpu_layers` | -1 | -1 = all on GPU, 0 = CPU only, N = specific layers |
| `temperature` | 1.0 | Higher = more random, lower = more focused |
| `max_tokens` | 1000 | Maximum output length (-1 = unlimited) |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 0 | Set > 0 for reproducible outputs |
| `n_ctx` | 8192 | Context window size (model dependent) |
| `unload_model` | True | Free memory after generation |

### Optional Inputs

| Parameter | Description |
|-----------|-------------|
| `images` | Connect images for vision models |
| `clip_model` | Select mmproj file for vision support |

## 5. Performance Tuning

### For Speed

- Set `unload_model=False` when running multiple generations
- Use Q4_K_M quantization (good balance)
- Reduce `n_ctx` if you don't need large context
- Use smaller models (3B-7B)

### For Quality

- Use larger models (13B-70B if VRAM allows)
- Use Q5_K_M or Q8_0 quantization
- Lower `temperature` (0.7-0.9) for more focused outputs
- Increase `max_tokens` for longer responses

### For Memory Efficiency

- Set `n_gpu_layers` to partial offload (e.g., 32 instead of -1)
- Use aggressive quantization (Q4_K_M)
- Keep `unload_model=True`
- Reduce `n_ctx`

## 6. Common Issues

### "No models found"

**Problem:** No .gguf files in `models/LLM/` directory.

**Solution:** Download models and place in `ComfyUI/models/LLM/`

### Model runs on CPU despite `n_gpu_layers=-1`

**Problem:** llama-cpp-python not compiled with CUDA.

**Solution:** Reinstall with CUDA support:
```bash
uv pip uninstall llama-cpp-python
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### CUDA Out of Memory

**Problem:** Model too large for GPU.

**Solutions:**
- Reduce `n_gpu_layers` (try 32, 24, 16)
- Use smaller model or more aggressive quantization
- Reduce `n_ctx`

### Vision model not working

**Problem:** Images provided but no output or errors.

**Solutions:**
- Ensure `clip_model` is selected
- Verify both model and mmproj are from same model family
- Check that model supports vision (Llava, Bakllava, etc.)

## 7. Tips & Tricks

### Prompt Engineering

Good prompts are clear and specific:

❌ Bad: "Tell me about art"
✅ Good: "Explain the key characteristics of Renaissance art in 3 paragraphs"

### Chaining with Other Nodes

```
[TextTemplate] → [LlamaCppGenerate] → [TextExtract] → [SaveText]
```

### Batching Images

The node supports multiple images - connect a batch:
```
[Load Image Batch] → images → [LlamaCppGenerate]
                               ↓ prompt: "Describe each image"
```

### Using with Segments

For image segmentation workflows:
```
[SEGSCaptioner] → segments → [LlamaCppGenerate] → captions
```

## 8. Next Steps

- Explore different models from [Hugging Face GGUF collection](https://huggingface.co/models?library=gguf)
- Experiment with quantization levels (Q4 vs Q5 vs Q8)
- Try vision models for image analysis
- Combine with other ComfyUI nodes for complex workflows

## Resources

- [Installation Guide](LLAMACPP_INSTALL.md) - Detailed installation instructions
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) - Upstream project
- [GGUF Models](https://huggingface.co/models?library=gguf) - Model downloads

## Need Help?

Check the [Installation Guide](LLAMACPP_INSTALL.md) troubleshooting section or open an issue on GitHub.
