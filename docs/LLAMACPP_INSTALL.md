# LlamaCpp CUDA Installation Guide

This guide covers installation of `llama-cpp-python` with CUDA support for GPU acceleration.

## Quick Start (Recommended)

### CUDA 12.1 (Most Common)

```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Verify Installation

```bash
python -c "from llama_cpp import Llama; print('✓ llama-cpp-python installed successfully')"
```

### Check CUDA Support

```bash
python -c "from llama_cpp import llama_cpp; print('CUDA available:', hasattr(llama_cpp, 'llama_supports_gpu_offload') and llama_cpp.llama_supports_gpu_offload())"
```

## Installation Options

### Option 1: Official Prebuilt Wheels (Recommended)

Official wheels are available for different CUDA versions:

**CUDA 12.4:**
```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**CUDA 12.2:**
```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

**CUDA 12.1:**
```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**CUDA 11.8:**
```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

### Option 2: Community Wheels (jllllll)

Community-maintained wheels with additional CPU optimizations (AVX2/AVX512):

**CUDA 12.1 + AVX2:**
```bash
uv pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121
```

**CUDA 12.1 + AVX512:**
```bash
uv pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX512/cu121
```

See [jllllll's repository](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels) for other versions.

### Option 3: CPU Only

If you don't have a CUDA-capable GPU or want CPU-only inference:

```bash
uv pip install llama-cpp-python
```

### Option 4: Build from Source (Advanced)

For maximum performance tuned to your specific hardware:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache
```

Additional build options:
- Add `-DGGML_CUDA_F16=on` for FP16 support
- Add `-DGGML_NATIVE=on` for CPU architecture optimizations

## Detecting Your CUDA Version

### Using nvcc

```bash
nvcc --version
```

Look for the line like `release 12.1, V12.1.105`.

### Using nvidia-smi

```bash
nvidia-smi
```

Look for the CUDA version in the top right corner.

## Troubleshooting

### Import Error: Cannot find libcuda.so

**Problem:** CUDA libraries not in system path.

**Solution:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Add to `~/.bashrc` for permanent fix.

### GPU Offloading Not Working

**Problem:** Model still runs on CPU despite `n_gpu_layers=-1`.

**Check CUDA availability:**
```python
from llama_cpp import llama_cpp
print(hasattr(llama_cpp, 'llama_supports_gpu_offload'))
print(llama_cpp.llama_supports_gpu_offload() if hasattr(llama_cpp, 'llama_supports_gpu_offload') else False)
```

If returns `False`, llama-cpp-python was not compiled with CUDA support. Reinstall with correct CUDA wheel.

### CUDA Out of Memory (OOM)

**Problem:** Model too large for GPU VRAM.

**Solutions:**
1. Reduce `n_gpu_layers` (e.g., from `-1` to `32`)
2. Use smaller quantized model (e.g., Q4_K_M instead of Q8_0)
3. Reduce `n_ctx` (context window size)
4. Enable `unload_model=True` to free memory after generation

### Wrong CUDA Version

**Problem:** Installed wheel for different CUDA version.

**Solution:** Uninstall and reinstall with correct version:
```bash
uv pip uninstall llama-cpp-python
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

## Model Recommendations

### For GPU (VRAM-based)

| VRAM | Recommended Model Size | Quantization |
|------|------------------------|--------------|
| 4-6 GB | 3B-7B | Q4_K_M |
| 8-12 GB | 7B-13B | Q4_K_M, Q5_K_M |
| 16-24 GB | 13B-34B | Q4_K_M, Q5_K_M |
| 24+ GB | 34B-70B | Q5_K_M, Q8_0 |

### For CPU (RAM-based)

CPU inference is slower but works with larger models if you have sufficient RAM:

| RAM | Recommended Model Size | Quantization |
|-----|------------------------|--------------|
| 8 GB | 3B | Q4_K_M |
| 16 GB | 7B | Q4_K_M |
| 32 GB | 13B-34B | Q4_K_M |
| 64+ GB | 34B-70B | Q5_K_M |

## Performance Tips

1. **Use quantized models**: Q4_K_M offers best balance of quality/speed
2. **Enable GPU layers**: Set `n_gpu_layers=-1` for full GPU acceleration
3. **Optimize context**: Don't use larger `n_ctx` than needed
4. **Keep model loaded**: Set `unload_model=False` when running multiple generations
5. **Use appropriate batch size**: For vision models, process images in batches

## Additional Resources

- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [GGUF Model Hub](https://huggingface.co/models?library=gguf)
- [Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)

## Need Help?

If you encounter issues:
1. Check this troubleshooting guide
2. Verify CUDA installation: `nvidia-smi`
3. Check llama-cpp-python version: `pip show llama-cpp-python`
4. Report issues at [GitHub Issues](https://github.com/abetlen/llama-cpp-python/issues)
