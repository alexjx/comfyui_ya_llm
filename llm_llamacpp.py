import base64
import gc
import importlib
import logging
import os
import sys
import traceback
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from llama_cpp import Llama, llama_cpp
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler,
    Llava16ChatHandler,
    Qwen25VLChatHandler,
)

import comfy.model_management
import folder_paths

# Set the logging level for httpx and httpcore to WARNING or ERROR
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Create logger for this module
logger = logging.getLogger(__name__)

# Llamacpp Module - Using fixed llama-cpp-python fork with Qwen3-VL support

# Register LLM folder if not already registered
llm_dir = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir, exist_ok=True)
folder_paths.add_model_folder_path("LLM", llm_dir)

# Global model cache
_model_cache = {}


def check_cuda_availability():
    """Check if llama-cpp-python was compiled with CUDA support"""
    try:
        # Check if CUDA backend is available
        has_cuda = (
            hasattr(llama_cpp, "llama_supports_gpu_offload")
            and llama_cpp.llama_supports_gpu_offload()
        )
        return has_cuda
    except Exception as e:
        logger.warning(f"Could not check CUDA availability: {e}")
        return False


# Check CUDA availability at module level
_CUDA_AVAILABLE = check_cuda_availability()
if not _CUDA_AVAILABLE:
    logger.warning(
        "llama-cpp-python was not compiled with CUDA support. "
        "GPU acceleration will be disabled. "
        "Install with: uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
    )
else:
    logger.info("llama-cpp-python CUDA support detected")


def clear_memory_for_llamacpp():
    """
    Aggressively clear ComfyUI and PyTorch memory before invoking llama.cpp.
    This is useful when llama.cpp runs on the same machine and may need GPU memory.
    """
    # Unload all ComfyUI models
    comfy.model_management.unload_all_models()

    # Force Python garbage collection
    gc.collect()

    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gguf_models():
    """Get list of GGUF model files from models/LLM directory"""
    try:
        files = folder_paths.get_filename_list("LLM")
        # Filter for .gguf files only
        gguf_files = [f for f in files if f.lower().endswith(".gguf")]
        return gguf_files if gguf_files else ["(no models found)"]
    except Exception as e:
        logger.error(f"Error listing GGUF models: {e}")
        return ["(no models found)"]


def get_mmproj_models():
    """Get list of mmproj (CLIP) files for vision models"""
    try:
        files = folder_paths.get_filename_list("LLM")
        # Filter for mmproj files (various naming patterns)
        mmproj_files = [
            f for f in files if "mmproj" in f.lower() and f.lower().endswith(".gguf")
        ]
        return ["(none)"] + mmproj_files
    except Exception as e:
        logger.error(f"Error listing mmproj models: {e}")
        return ["(none)"]


def convert_image_to_data_url(img_tensor):
    """Convert ComfyUI IMAGE tensor to data URL for llama.cpp"""
    # img_tensor shape: [H, W, C], range [0, 1]
    img_data = 255.0 * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"


def get_or_load_model(model_path, n_ctx, n_gpu_layers, projector_path, has_images):
    """Get model from cache or load it"""
    cache_key = f"{model_path}_{n_ctx}_{n_gpu_layers}_{projector_path}"

    if cache_key in _model_cache:
        logger.info(f"Reusing cached model: {os.path.basename(model_path)}")
        return _model_cache[cache_key]

    # Convert to absolute path if needed
    model_path = os.path.abspath(model_path)
    if projector_path:
        projector_path = os.path.abspath(projector_path)

    logger.info(f"Loading model: {os.path.basename(model_path)}")

    # Warn if GPU requested but CUDA not available
    if n_gpu_layers != 0 and not _CUDA_AVAILABLE:
        logger.warning(
            f"GPU layers requested (n_gpu_layers={n_gpu_layers}) but CUDA not available. "
            f"Falling back to CPU. Install llama-cpp-python with CUDA support for GPU acceleration."
        )
        n_gpu_layers = 0  # Force CPU

    # Prepare kwargs
    kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": False,
    }

    # Vision support
    if projector_path:
        if not os.path.exists(projector_path):
            logger.warning(f"Projector model not found: {projector_path}")
        else:
            try:
                # Force reload to avoid stale cache
                import llama_cpp.llama_chat_format

                importlib.reload(llama_cpp.llama_chat_format)

                # Auto-detect model type
                model_name = os.path.basename(model_path).lower()
                projector_name = os.path.basename(projector_path).lower()

                if "qwen" in model_name or "qwen" in projector_name:
                    chat_handler = Qwen25VLChatHandler(
                        clip_model_path=projector_path, verbose=False
                    )
                    logger.info("Using Qwen vision chat handler")
                elif "1.6" in projector_name:
                    chat_handler = Llava16ChatHandler(
                        clip_model_path=projector_path, verbose=False
                    )
                    logger.info("Using Llava 1.6 chat handler")
                else:
                    chat_handler = Llava15ChatHandler(
                        clip_model_path=projector_path, verbose=False
                    )
                    logger.info("Using Llava 1.5 chat handler")

                kwargs["chat_handler"] = chat_handler
                kwargs["logits_all"] = True
            except Exception as e:
                logger.error(f"Failed to load vision model handler: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
    elif has_images:
        logger.warning("Images provided but no projector specified. Vision disabled.")

    # Load model
    try:
        # Suppress llama.cpp stderr output (CUDA init, layer loading, etc.)
        # But allow errors through for debugging
        if has_images:
            # Don't suppress for vision models - need to see mtmd errors
            model = Llama(**kwargs)
        else:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")
            try:
                model = Llama(**kwargs)
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr

        _model_cache[cache_key] = model

        # Log actual GPU usage
        if n_gpu_layers > 0:
            logger.info(f"Model loaded with {n_gpu_layers} GPU layers")
        else:
            logger.info("Model loaded (CPU only)")

        return model
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg)


def unload_model_from_cache(cache_key):
    """Explicitly unload a model from cache and free memory"""
    if cache_key in _model_cache:
        logger.info("Unloading model from cache")

        # Suppress llama.cpp cleanup output
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

        try:
            del _model_cache[cache_key]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
            torch.cuda.synchronize()


class LlamacppGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "What is Art?"}),
                "model": (
                    get_gguf_models(),
                    {"tooltip": "GGUF model from models/LLM directory"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 2, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1000,
                        "min": -1,
                        "max": 8192,
                        "tooltip": "-1 for unlimited",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 128,
                        "max": 32768,
                        "tooltip": "Context window size",
                    },
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 256,
                        "tooltip": "-1 for all layers on GPU, 0 for CPU only. Requires llama-cpp-python with CUDA support.",
                    },
                ),
                "unload_model": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload model after generation to free memory",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "forceInput": True,
                        "tooltip": "Images for vision models (requires projector)",
                    },
                ),
                "projector": (
                    get_mmproj_models(),
                    {"tooltip": "Vision projection model for multimodal models (mmproj*.gguf)"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "llamacpp_generate"
    CATEGORY = "Yet Another LLM"

    def llamacpp_generate(
        self,
        prompt,
        model,
        seed,
        temperature,
        max_tokens,
        n_ctx,
        n_gpu_layers,
        unload_model,
        images=None,
        projector="(none)",
    ):
        # 1. Validate model
        if model == "(no models found)":
            return (
                "Error: No GGUF models found in models/LLM directory. Please add models.",
            )

        model_path = folder_paths.get_full_path("LLM", model)
        if not model_path or not os.path.exists(model_path):
            return (f"Error: Model file not found: {model}",)

        # 2. Resolve projector path (if provided)
        projector_path = ""
        if projector and projector != "(none)":
            projector_path = folder_paths.get_full_path("LLM", projector)
            if not projector_path or not os.path.exists(projector_path):
                logger.warning(f"Projector model not found: {projector}")
                projector_path = ""

        # 3. Clear ComfyUI memory
        clear_memory_for_llamacpp()

        # 4. Get or load model from cache
        cache_key = f"{model_path}_{n_ctx}_{n_gpu_layers}_{projector_path}"
        try:
            llm_model = get_or_load_model(
                model_path, n_ctx, n_gpu_layers, projector_path, images is not None
            )
        except Exception as e:
            return (f"Error loading model: {str(e)}",)

        # 5. Build messages
        if images is not None:
            content = [{"type": "text", "text": prompt}]
            # Process all images in batch
            for img in images:
                img_data_url = convert_image_to_data_url(img)
                content.append(
                    {"type": "image_url", "image_url": {"url": img_data_url}}
                )
            messages = [{"role": "user", "content": content}]
            logger.info(f"Processing {len(images)} image(s) with prompt")
        else:
            messages = [{"role": "user", "content": prompt}]

        # 6. Generate with streaming output
        full_response = ""
        print(f"Llamacpp generation ({os.path.basename(model)}):")
        # print(f"\033[33mPrompt: {prompt}\033[0m")
        # print()  # Newline between prompt and response

        try:
            stream = llm_model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed if seed > 0 else None,
                stream=True,
            )

            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    full_response += delta
                    print(f"\033[32m{delta}\033[0m", end="", flush=True)

            print()  # Newline after generation

            # Get timing statistics from llama.cpp
            timings = llama_cpp.llama_perf_context(llm_model._ctx.ctx)
            prompt_tokens = timings.n_p_eval
            completion_tokens = timings.n_eval
            total_time_s = (timings.t_p_eval_ms + timings.t_eval_ms) / 1000.0

            # Display statistics similar to Ollama format
            if completion_tokens > 0 and timings.t_eval_ms > 0:
                tokens_per_sec = completion_tokens / (timings.t_eval_ms / 1000.0)
                logger.info(
                    f"Generated: prompt {prompt_tokens} tokens - response {completion_tokens} tokens in {total_time_s:.2f}s ({tokens_per_sec:.2f} tokens/s)"
                )

        except torch.cuda.OutOfMemoryError:
            error = "CUDA OOM. Try reducing n_gpu_layers or use smaller model."
            logger.error(error)
            if unload_model:
                unload_model_from_cache(cache_key)
            return (error,)
        except Exception as e:
            error = f"Llamacpp Error: {type(e).__name__}: {str(e)}"
            logger.error(error)
            if unload_model:
                unload_model_from_cache(cache_key)
            return (error,)

        # 7. Handle unload
        if unload_model:
            unload_model_from_cache(cache_key)
        else:
            logger.info("Model kept in cache for future use")

        return (full_response.strip(),)


NODE_CLASS_MAPPINGS = {
    "yaLlamacppGenerate": LlamacppGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaLlamacppGenerate": "Llamacpp Generate",
}
