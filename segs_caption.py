import base64
import logging
import os
import re
from io import BytesIO

import jinja2
import numpy as np
import ollama
import torch
from PIL import Image
from llama_cpp import llama_cpp

import folder_paths
from .llm_llamacpp import (
    clear_memory_for_llamacpp,
    convert_image_to_data_url,
    get_gguf_models,
    get_mmproj_models,
    get_or_load_model,
    unload_model_from_cache,
)
from .llm_ollama import clear_memory_for_ollama, wait_for_model_unload


THINKING_OPTIONS = ["ON", "OFF", "HIGH", "MEDIUM", "LOW", "NONE"]


def map_thinking(thinking):
    if thinking == "OFF":
        return False
    if thinking == "ON":
        return True
    if thinking == "HIGH":
        return "high"
    if thinking == "MEDIUM":
        return "medium"
    if thinking == "LOW":
        return "low"
    if thinking == "NONE":
        return None
    return None


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def format_caption(caption, reasoning, keep_reason):
    if keep_reason:
        if reasoning:
            caption = f"<think>{reasoning}</think>\n{caption}"
    else:
        caption = strip_thinking(caption)
    return caption.strip().replace("\n", " ").replace("\r", " ")


class SEGSCaptioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
                "prompt_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the major item briefly in one paragraph with simple and concise wording:",
                    },
                ),
                "url": (
                    "STRING",
                    {"multiline": False, "default": "http://127.0.0.1:11434"},
                ),
                "model": ((), {}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 2, "step": 0.05},
                ),
                "num_ctx": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "num_predict": (
                    "INT",
                    {"default": 500, "min": -1, "max": 4096, "step": 1},
                ),
                "keep_alive": (
                    "INT",
                    {"default": 0, "min": -1, "max": 3600, "step": 1},
                ),
                "thinking": (THINKING_OPTIONS, {"default": "NONE"}),
                "keep_reason": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "fallback_image_opt": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("wildcard_captions",)
    FUNCTION = "caption_segs"
    CATEGORY = "Yet Another LLM"

    def segs_scale_match(self, segs, target_shape):
        """Scale SEGS crop_regions to match target image shape"""
        from collections import namedtuple

        SEG = namedtuple(
            "SEG",
            [
                "cropped_image",
                "cropped_mask",
                "confidence",
                "crop_region",
                "bbox",
                "label",
                "control_net_wrapper",
            ],
            defaults=[None],
        )

        h = segs[0][0]
        w = segs[0][1]

        th = target_shape[1]
        tw = target_shape[2]

        if (h == th and w == tw) or h == 0 or w == 0:
            return segs

        rh = th / h
        rw = tw / w

        new_segs = []
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            crop_region = int(x1 * rw), int(y1 * rh), int(x2 * rw), int(y2 * rh)
            bbox = int(bx1 * rw), int(by1 * rh), int(bx2 * rw), int(by2 * rh)

            # Keep cropped_mask and cropped_image as is (we'll crop from fallback anyway)
            new_seg = SEG(
                seg.cropped_image,
                seg.cropped_mask,
                seg.confidence,
                crop_region,
                bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        return (th, tw), new_segs

    def render_prompt(
        self, template, seg, index, total_segments, image_tensor, crop_region
    ):
        """Render Jinja2 template with seg context relative to cropped image"""
        # Get cropped image dimensions
        height, width = image_tensor.shape[0], image_tensor.shape[1]

        # Calculate bbox relative to the cropped region
        # bbox is in original image coords, crop_region defines the crop
        x1_crop, y1_crop, x2_crop, y2_crop = crop_region
        bx1, by1, bx2, by2 = seg.bbox

        # Convert to crop-relative coordinates
        bbox_rel = (
            bx1 - x1_crop,
            by1 - y1_crop,
            bx2 - x1_crop,
            by2 - y1_crop,
        )

        # Normalized bbox (0.0 to 1.0)
        bbox_norm = (
            bbox_rel[0] / width if width > 0 else 0,
            bbox_rel[1] / height if height > 0 else 0,
            bbox_rel[2] / width if width > 0 else 0,
            bbox_rel[3] / height if height > 0 else 0,
        )

        context = {
            "label": seg.label if seg.label else "",
            "index": index,
            "total_segments": total_segments,
            "confidence": seg.confidence,
            "width": width,
            "height": height,
            "bbox": bbox_rel,
            "bbox_norm": bbox_norm,
            "has_label": bool(seg.label),
        }

        # Check if template contains Jinja2 syntax
        if not re.search(r"\{[{%#]", template):
            # No Jinja2 syntax, return as-is (backward compatible)
            return template

        # Render with Jinja2
        try:
            env = jinja2.Environment(autoescape=False)
            tpl = env.from_string(template)
            return tpl.render(**context)
        except jinja2.exceptions.TemplateError as e:
            # Fail the node on template errors
            raise ValueError(
                f"Template rendering failed for segment {index}: {str(e)}. "
                f"Available variables: {', '.join(sorted(context.keys()))}"
            )

    def caption_segs(
        self,
        segs,
        prompt_template,
        url,
        model,
        seed,
        temperature,
        num_ctx,
        num_predict,
        keep_alive,
        thinking,
        keep_reason,
        fallback_image_opt=None,
    ):
        # Extract SEGS structure
        shape, seg_list = segs

        if len(seg_list) == 0:
            logging.warning("SEGSCaptioner: Empty SEGS list provided")
            return ("[LAB]\n",)

        # Scale segs to match fallback image if provided (like SEGSPreview does)
        if fallback_image_opt is not None:
            segs = self.segs_scale_match(segs, fallback_image_opt.shape)
            shape, seg_list = segs

        # Initialize Ollama client
        client = ollama.Client(host=url)
        model = model.strip()
        thinking = map_thinking(thinking)

        # Options for Ollama
        options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

        captions = []

        # Optimize keep_alive for multiple segments
        has_multiple_segs = len(seg_list) > 1

        logging.info(
            f"SEGSCaptioner: Processing {len(seg_list)} segments with model {model}"
        )

        for idx, seg in enumerate(seg_list, start=1):
            label = str(idx)

            # Determine which image to use
            image_tensor = None

            if seg.cropped_image is not None:
                # Use the seg's cropped image
                image_tensor = seg.cropped_image
                if len(image_tensor.shape) == 4:
                    # Take first image from batch if batched
                    image_tensor = image_tensor[0]
            elif fallback_image_opt is not None:
                # Crop from fallback image using seg's crop_region (like SEGSPreview does)
                ref_image = (
                    fallback_image_opt[0].unsqueeze(0)
                    if len(fallback_image_opt.shape) == 4
                    else fallback_image_opt.unsqueeze(0)
                )
                x1, y1, x2, y2 = seg.crop_region
                cropped_image = ref_image[:, y1:y2, x1:x2, :]
                image_tensor = cropped_image[0]  # Remove batch dimension
                logging.info(
                    f"SEGSCaptioner: Seg {idx} cropped from fallback image using crop_region {seg.crop_region}"
                )
            else:
                # No image available
                logging.warning(
                    f"SEGSCaptioner: Seg {idx} has no cropped_image and no fallback provided"
                )
                captions.append(f"[{label}] (no image)")
                continue

            # Convert tensor to base64
            try:
                i = 255.0 * image_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                logging.error(
                    f"SEGSCaptioner: Failed to convert seg {idx} to image: {e}"
                )
                captions.append(f"[{label}] (image conversion failed)")
                continue

            # Render prompt with seg context
            try:
                rendered_prompt = self.render_prompt(
                    prompt_template,
                    seg,
                    idx,
                    len(seg_list),
                    image_tensor,
                    seg.crop_region,
                )
            except ValueError as e:
                logging.error(f"SEGSCaptioner: {e}")
                raise

            # Clear memory before Ollama call
            clear_memory_for_ollama()

            # Call Ollama to caption the image
            try:
                print(f"\n\033[36m[Seg {idx}/{len(seg_list)}]\033[0m", flush=True)
                print(f"\033[33mPrompt: {rendered_prompt}\033[0m", flush=True)
                print("\033[32m", end="", flush=True)

                options["seed"] = seed + idx - 1  # Increment seed per seg

                # Keep model loaded for intermediate segments, unload on last
                is_last_seg = idx == len(seg_list)
                effective_keep_alive = (
                    0 if is_last_seg else (1 if has_multiple_segs else keep_alive)
                )

                stream = client.generate(
                    model=model,
                    prompt=rendered_prompt,
                    images=[img_b64],
                    options=options,
                    keep_alive=f"{effective_keep_alive}s",
                    think=thinking,  # type: ignore
                    stream=True,
                )

                caption = ""
                reasoning = ""
                for chunk in stream:
                    if "thinking" in chunk:
                        think_text = chunk["thinking"]
                        print(f"\033[33m{think_text}\033[0m", end="", flush=True)
                        reasoning += think_text
                    if "response" in chunk:
                        response_text = chunk["response"]
                        print(f"\033[32m{response_text}\033[0m", end="", flush=True)
                        caption += response_text
                    if "done" in chunk and chunk["done"]:
                        print()  # New line after completion
                        logging.info(
                            f"SEGSCaptioner seg {idx}: {chunk.get('prompt_eval_count', 0)} prompt tokens, "
                            f"{chunk.get('eval_count', 0)} response tokens in "
                            f"{chunk.get('total_duration', 0) / (10**9):.2f}s "
                            f"({chunk.get('eval_count', 0) / chunk.get('eval_duration', 1) * (10**9):.2f} tokens/s)"
                        )
                        break

                caption = format_caption(caption, reasoning, keep_reason)

                captions.append(f"[{label}] {caption}")

            except Exception as e:
                logging.error(f"SEGSCaptioner: Failed to caption seg {idx}: {e}")
                captions.append(f"[{label}] (captioning failed)")

        # Format output as [LAB] wildcard format
        wildcard_output = "[LAB]\n" + "\n".join(captions)

        logging.info(f"SEGSCaptioner: Completed captioning {len(seg_list)} segments")

        # Wait for model to unload if we processed multiple segments
        if has_multiple_segs:
            logging.info("SEGSCaptioner: Waiting for Ollama model to unload...")
            wait_for_model_unload(client, model)
            clear_memory_for_ollama()

        return (wildcard_output,)


class SEGSCaptionerV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
                "prompt_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the major item briefly in one paragraph with simple and concise wording:",
                    },
                ),
                "url": (
                    "STRING",
                    {"multiline": False, "default": "http://127.0.0.1:11434"},
                ),
                "model": ((), {}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 2, "step": 0.05},
                ),
                "num_ctx": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "num_predict": (
                    "INT",
                    {"default": 500, "min": -1, "max": 4096, "step": 1},
                ),
                "keep_alive": (
                    "INT",
                    {"default": 0, "min": -1, "max": 3600, "step": 1},
                ),
                "thinking": (THINKING_OPTIONS, {"default": "NONE"}),
                "keep_reason": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "fallback_image_opt": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "caption_segs"
    CATEGORY = "Yet Another LLM"
    OUTPUT_IS_LIST = (True,)

    def segs_scale_match(self, segs, target_shape):
        """Scale SEGS crop_regions to match target image shape"""
        from collections import namedtuple

        SEG = namedtuple(
            "SEG",
            [
                "cropped_image",
                "cropped_mask",
                "confidence",
                "crop_region",
                "bbox",
                "label",
                "control_net_wrapper",
            ],
            defaults=[None],
        )

        h = segs[0][0]
        w = segs[0][1]

        th = target_shape[1]
        tw = target_shape[2]

        if (h == th and w == tw) or h == 0 or w == 0:
            return segs

        rh = th / h
        rw = tw / w

        new_segs = []
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            crop_region = int(x1 * rw), int(y1 * rh), int(x2 * rw), int(y2 * rh)
            bbox = int(bx1 * rw), int(by1 * rh), int(bx2 * rw), int(by2 * rh)

            new_seg = SEG(
                seg.cropped_image,
                seg.cropped_mask,
                seg.confidence,
                crop_region,
                bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        return (th, tw), new_segs

    def render_prompt(
        self, template, seg, index, total_segments, image_tensor, crop_region
    ):
        """Render Jinja2 template with seg context relative to cropped image"""
        # Get cropped image dimensions
        height, width = image_tensor.shape[0], image_tensor.shape[1]

        # Calculate bbox relative to the cropped region
        # bbox is in original image coords, crop_region defines the crop
        x1_crop, y1_crop, x2_crop, y2_crop = crop_region
        bx1, by1, bx2, by2 = seg.bbox

        # Convert to crop-relative coordinates
        bbox_rel = (
            bx1 - x1_crop,
            by1 - y1_crop,
            bx2 - x1_crop,
            by2 - y1_crop,
        )

        # Normalized bbox (0.0 to 1.0)
        bbox_norm = (
            bbox_rel[0] / width if width > 0 else 0,
            bbox_rel[1] / height if height > 0 else 0,
            bbox_rel[2] / width if width > 0 else 0,
            bbox_rel[3] / height if height > 0 else 0,
        )

        context = {
            "label": seg.label if seg.label else "",
            "index": index,
            "total_segments": total_segments,
            "confidence": seg.confidence,
            "width": width,
            "height": height,
            "bbox": bbox_rel,
            "bbox_norm": bbox_norm,
            "has_label": bool(seg.label),
        }

        # Check if template contains Jinja2 syntax
        if not re.search(r"\{[{%#]", template):
            # No Jinja2 syntax, return as-is (backward compatible)
            return template

        # Render with Jinja2
        try:
            env = jinja2.Environment(autoescape=False)
            tpl = env.from_string(template)
            return tpl.render(**context)
        except jinja2.exceptions.TemplateError as e:
            # Fail the node on template errors
            raise ValueError(
                f"Template rendering failed for segment {index}: {str(e)}. "
                f"Available variables: {', '.join(sorted(context.keys()))}"
            )

    def caption_segs(
        self,
        segs,
        prompt_template,
        url,
        model,
        seed,
        temperature,
        num_ctx,
        num_predict,
        keep_alive,
        thinking,
        keep_reason,
        fallback_image_opt=None,
    ):
        shape, seg_list = segs

        if len(seg_list) == 0:
            logging.warning("SEGSCaptionerV2: Empty SEGS list provided")
            return ([],)

        if fallback_image_opt is not None:
            segs = self.segs_scale_match(segs, fallback_image_opt.shape)
            shape, seg_list = segs

        client = ollama.Client(host=url)
        model = model.strip()
        thinking = map_thinking(thinking)

        options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

        captions = []

        # Optimize keep_alive for multiple segments
        has_multiple_segs = len(seg_list) > 1

        logging.info(
            f"SEGSCaptionerV2: Processing {len(seg_list)} segments with model {model}"
        )

        for idx, seg in enumerate(seg_list, start=1):
            image_tensor = None

            if seg.cropped_image is not None:
                image_tensor = seg.cropped_image
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]
            elif fallback_image_opt is not None:
                ref_image = (
                    fallback_image_opt[0].unsqueeze(0)
                    if len(fallback_image_opt.shape) == 4
                    else fallback_image_opt.unsqueeze(0)
                )
                x1, y1, x2, y2 = seg.crop_region
                cropped_image = ref_image[:, y1:y2, x1:x2, :]
                image_tensor = cropped_image[0]
                logging.info(
                    f"SEGSCaptionerV2: Seg {idx} cropped from fallback image using crop_region {seg.crop_region}"
                )
            else:
                logging.warning(
                    f"SEGSCaptionerV2: Seg {idx} has no cropped_image and no fallback provided"
                )
                captions.append("(no image)")
                continue

            try:
                i = 255.0 * image_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                logging.error(
                    f"SEGSCaptionerV2: Failed to convert seg {idx} to image: {e}"
                )
                captions.append("(image conversion failed)")
                continue

            # Render prompt with seg context
            try:
                rendered_prompt = self.render_prompt(
                    prompt_template,
                    seg,
                    idx,
                    len(seg_list),
                    image_tensor,
                    seg.crop_region,
                )
            except ValueError as e:
                logging.error(f"SEGSCaptionerV2: {e}")
                raise

            clear_memory_for_ollama()

            try:
                print(f"\n\033[36m[Seg {idx}/{len(seg_list)}]\033[0m", flush=True)
                print(f"\033[33mPrompt: {rendered_prompt}\033[0m", flush=True)
                print("\033[32m", end="", flush=True)

                options["seed"] = seed + idx - 1

                # Keep model loaded for intermediate segments, unload on last
                is_last_seg = idx == len(seg_list)
                effective_keep_alive = (
                    0 if is_last_seg else (1 if has_multiple_segs else keep_alive)
                )

                stream = client.generate(
                    model=model,
                    prompt=rendered_prompt,
                    images=[img_b64],
                    options=options,
                    keep_alive=f"{effective_keep_alive}s",
                    think=thinking,  # type: ignore
                    stream=True,
                )

                caption = ""
                reasoning = ""
                for chunk in stream:
                    if "thinking" in chunk:
                        think_text = chunk["thinking"]
                        print(f"\033[33m{think_text}\033[0m", end="", flush=True)
                        reasoning += think_text
                    if "response" in chunk:
                        response_text = chunk["response"]
                        print(f"\033[32m{response_text}\033[0m", end="", flush=True)
                        caption += response_text
                    if "done" in chunk and chunk["done"]:
                        print()  # New line after completion
                        logging.info(
                            f"SEGSCaptionerV2 seg {idx}: {chunk.get('prompt_eval_count', 0)} prompt tokens, "
                            f"{chunk.get('eval_count', 0)} response tokens in "
                            f"{chunk.get('total_duration', 0) / (10**9):.2f}s "
                            f"({chunk.get('eval_count', 0) / chunk.get('eval_duration', 1) * (10**9):.2f} tokens/s)"
                        )
                        break

                caption = format_caption(caption, reasoning, keep_reason)
                captions.append(caption)

            except Exception as e:
                logging.error(f"SEGSCaptionerV2: Failed to caption seg {idx}: {e}")
                captions.append("(captioning failed)")

        logging.info(f"SEGSCaptionerV2: Completed captioning {len(seg_list)} segments")

        # Wait for model to unload if we processed multiple segments
        if has_multiple_segs:
            logging.info("SEGSCaptionerV2: Waiting for Ollama model to unload...")
            wait_for_model_unload(client, model)
            clear_memory_for_ollama()

        return (captions,)


class LlamacppSEGSCaptioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
                "prompt_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the major item briefly in one paragraph with simple and concise wording:",
                    },
                ),
                "model": (
                    get_gguf_models(),
                    {"tooltip": "GGUF model from models/LLM directory"},
                ),
                "projector": (
                    [f for f in get_mmproj_models() if f != "(none)"],
                    {"tooltip": "Vision projection model for multimodal models (mmproj*.gguf) - REQUIRED for image captioning"},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 2, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 500,
                        "min": -1,
                        "max": 4096,
                        "tooltip": "-1 for unlimited",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 2048,
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
                        "tooltip": "-1 for all layers on GPU, 0 for CPU only",
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
                "fallback_image_opt": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("wildcard_captions",)
    FUNCTION = "caption_segs"
    CATEGORY = "Yet Another LLM"

    def segs_scale_match(self, segs, target_shape):
        """Scale SEGS crop_regions to match target image shape"""
        from collections import namedtuple

        SEG = namedtuple(
            "SEG",
            [
                "cropped_image",
                "cropped_mask",
                "confidence",
                "crop_region",
                "bbox",
                "label",
                "control_net_wrapper",
            ],
            defaults=[None],
        )

        h = segs[0][0]
        w = segs[0][1]

        th = target_shape[1]
        tw = target_shape[2]

        if (h == th and w == tw) or h == 0 or w == 0:
            return segs

        rh = th / h
        rw = tw / w

        new_segs = []
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            crop_region = int(x1 * rw), int(y1 * rh), int(x2 * rw), int(y2 * rh)
            bbox = int(bx1 * rw), int(by1 * rh), int(bx2 * rw), int(by2 * rh)

            # Keep cropped_mask and cropped_image as is (we'll crop from fallback anyway)
            new_seg = SEG(
                seg.cropped_image,
                seg.cropped_mask,
                seg.confidence,
                crop_region,
                bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        return (th, tw), new_segs

    def render_prompt(
        self, template, seg, index, total_segments, image_tensor, crop_region
    ):
        """Render Jinja2 template with seg context relative to cropped image"""
        # Get cropped image dimensions
        height, width = image_tensor.shape[0], image_tensor.shape[1]

        # Calculate bbox relative to the cropped region
        # bbox is in original image coords, crop_region defines the crop
        x1_crop, y1_crop, x2_crop, y2_crop = crop_region
        bx1, by1, bx2, by2 = seg.bbox

        # Convert to crop-relative coordinates
        bbox_rel = (
            bx1 - x1_crop,
            by1 - y1_crop,
            bx2 - x1_crop,
            by2 - y1_crop,
        )

        # Normalized bbox (0.0 to 1.0)
        bbox_norm = (
            bbox_rel[0] / width if width > 0 else 0,
            bbox_rel[1] / height if height > 0 else 0,
            bbox_rel[2] / width if width > 0 else 0,
            bbox_rel[3] / height if height > 0 else 0,
        )

        context = {
            "label": seg.label if seg.label else "",
            "index": index,
            "total_segments": total_segments,
            "confidence": seg.confidence,
            "width": width,
            "height": height,
            "bbox": bbox_rel,
            "bbox_norm": bbox_norm,
            "has_label": bool(seg.label),
        }

        # Check if template contains Jinja2 syntax
        if not re.search(r"\{[{%#]", template):
            # No Jinja2 syntax, return as-is (backward compatible)
            return template

        # Render with Jinja2
        try:
            env = jinja2.Environment(autoescape=False)
            tpl = env.from_string(template)
            return tpl.render(**context)
        except jinja2.exceptions.TemplateError as e:
            # Fail the node on template errors
            raise ValueError(
                f"Template rendering failed for segment {index}: {str(e)}. "
                f"Available variables: {', '.join(sorted(context.keys()))}"
            )

    def caption_segs(
        self,
        segs,
        prompt_template,
        model,
        projector,
        seed,
        temperature,
        max_tokens,
        n_ctx,
        n_gpu_layers,
        unload_model,
        fallback_image_opt=None,
    ):
        # Extract SEGS structure
        shape, seg_list = segs

        if len(seg_list) == 0:
            logging.warning("LlamacppSEGSCaptioner: Empty SEGS list provided")
            return ("[LAB]\n",)

        # Validate model
        if model == "(no models found)":
            return ("[LAB]\n(error: no models found)",)

        model_path = folder_paths.get_full_path("LLM", model)
        if not model_path or not os.path.exists(model_path):
            return (f"[LAB]\n(error: model file not found: {model})",)

        # Resolve projector path (required for image captioning)
        projector_path = folder_paths.get_full_path("LLM", projector)
        if not projector_path or not os.path.exists(projector_path):
            return (
                f"[LAB]\n(error: Projector model not found: {projector}. Vision projector required for image captioning.)",
            )

        # Scale segs to match fallback image if provided (like SEGSPreview does)
        if fallback_image_opt is not None:
            segs = self.segs_scale_match(segs, fallback_image_opt.shape)
            shape, seg_list = segs

        captions = []

        logging.info(
            f"LlamacppSEGSCaptioner: Processing {len(seg_list)} segments with model {model}"
        )

        # Load model once for all segments
        clear_memory_for_llamacpp()
        cache_key = f"{model_path}_{n_ctx}_{n_gpu_layers}_{projector_path}"

        try:
            llm_model = get_or_load_model(
                model_path, n_ctx, n_gpu_layers, projector_path, has_images=True
            )
        except Exception as e:
            logging.error(f"LlamacppSEGSCaptioner: Failed to load model: {e}")
            return (f"[LAB]\n(error: failed to load model: {str(e)})",)

        for idx, seg in enumerate(seg_list, start=1):
            label = str(idx)

            # Determine which image to use
            image_tensor = None

            if seg.cropped_image is not None:
                # Use the seg's cropped image
                image_tensor = seg.cropped_image
                if len(image_tensor.shape) == 4:
                    # Take first image from batch if batched
                    image_tensor = image_tensor[0]
            elif fallback_image_opt is not None:
                # Crop from fallback image using seg's crop_region (like SEGSPreview does)
                ref_image = (
                    fallback_image_opt[0].unsqueeze(0)
                    if len(fallback_image_opt.shape) == 4
                    else fallback_image_opt.unsqueeze(0)
                )
                x1, y1, x2, y2 = seg.crop_region
                cropped_image = ref_image[:, y1:y2, x1:x2, :]
                image_tensor = cropped_image[0]  # Remove batch dimension
                logging.info(
                    f"LlamacppSEGSCaptioner: Seg {idx} cropped from fallback image using crop_region {seg.crop_region}"
                )
            else:
                # No image available
                logging.warning(
                    f"LlamacppSEGSCaptioner: Seg {idx} has no cropped_image and no fallback provided"
                )
                captions.append(f"[{label}] (no image)")
                continue

            # Render prompt with seg context
            try:
                rendered_prompt = self.render_prompt(
                    prompt_template,
                    seg,
                    idx,
                    len(seg_list),
                    image_tensor,
                    seg.crop_region,
                )
            except ValueError as e:
                logging.error(f"LlamacppSEGSCaptioner: {e}")
                raise

            # Call Llamacpp to caption the image
            try:
                print(f"\n\033[36m[Seg {idx}/{len(seg_list)}]\033[0m", flush=True)
                print(f"\033[33mPrompt: {rendered_prompt}\033[0m", flush=True)
                print()

                # Build messages with image
                images_batch = image_tensor.unsqueeze(0)  # [1, H, W, C]
                content = [{"type": "text", "text": rendered_prompt}]
                for img in images_batch:
                    img_data_url = convert_image_to_data_url(img)
                    content.append(
                        {"type": "image_url", "image_url": {"url": img_data_url}}
                    )
                messages = [{"role": "user", "content": content}]

                # Generate with streaming
                stream = llm_model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed + idx - 1 if seed > 0 else None,  # Increment seed per seg
                    stream=True,
                )

                caption = ""
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        caption += delta
                        print(f"\033[32m{delta}\033[0m", end="", flush=True)

                print()  # Newline after completion

                # Get timing statistics from llama.cpp
                timings = llama_cpp.llama_perf_context(llm_model._ctx.ctx)
                prompt_tokens = timings.n_p_eval
                completion_tokens = timings.n_eval
                total_time_s = (timings.t_p_eval_ms + timings.t_eval_ms) / 1000.0

                if completion_tokens > 0 and timings.t_eval_ms > 0:
                    tokens_per_sec = completion_tokens / (timings.t_eval_ms / 1000.0)
                    logging.info(
                        f"LlamacppSEGSCaptioner seg {idx}: prompt {prompt_tokens} tokens - response {completion_tokens} tokens in {total_time_s:.2f}s ({tokens_per_sec:.2f} tokens/s)"
                    )

                caption = caption.strip().replace("\n", " ").replace("\r", " ")
                captions.append(f"[{label}] {caption}")

            except torch.cuda.OutOfMemoryError:
                error = "CUDA OOM. Try reducing n_gpu_layers or use smaller model."
                logging.error(f"LlamacppSEGSCaptioner: {error}")
                captions.append(f"[{label}] ({error})")
            except Exception as e:
                logging.error(
                    f"LlamacppSEGSCaptioner: Failed to caption seg {idx}: {e}"
                )
                captions.append(f"[{label}] (captioning failed)")

        # Format output as [LAB] wildcard format
        wildcard_output = "[LAB]\n" + "\n".join(captions)

        logging.info(
            f"LlamacppSEGSCaptioner: Completed captioning {len(seg_list)} segments"
        )

        # Unload model if requested
        if unload_model:
            unload_model_from_cache(cache_key)
            clear_memory_for_llamacpp()

        return (wildcard_output,)


class LlamacppSEGSCaptionerV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
                "prompt_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the major item briefly in one paragraph with simple and concise wording:",
                    },
                ),
                "model": (
                    get_gguf_models(),
                    {"tooltip": "GGUF model from models/LLM directory"},
                ),
                "projector": (
                    [f for f in get_mmproj_models() if f != "(none)"],
                    {"tooltip": "Vision projection model for multimodal models (mmproj*.gguf) - REQUIRED for image captioning"},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 2, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 500,
                        "min": -1,
                        "max": 4096,
                        "tooltip": "-1 for unlimited",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 2048,
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
                        "tooltip": "-1 for all layers on GPU, 0 for CPU only",
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
                "fallback_image_opt": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "caption_segs"
    CATEGORY = "Yet Another LLM"
    OUTPUT_IS_LIST = (True,)

    def segs_scale_match(self, segs, target_shape):
        """Scale SEGS crop_regions to match target image shape"""
        from collections import namedtuple

        SEG = namedtuple(
            "SEG",
            [
                "cropped_image",
                "cropped_mask",
                "confidence",
                "crop_region",
                "bbox",
                "label",
                "control_net_wrapper",
            ],
            defaults=[None],
        )

        h = segs[0][0]
        w = segs[0][1]

        th = target_shape[1]
        tw = target_shape[2]

        if (h == th and w == tw) or h == 0 or w == 0:
            return segs

        rh = th / h
        rw = tw / w

        new_segs = []
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            crop_region = int(x1 * rw), int(y1 * rh), int(x2 * rw), int(y2 * rh)
            bbox = int(bx1 * rw), int(by1 * rh), int(bx2 * rw), int(by2 * rh)

            new_seg = SEG(
                seg.cropped_image,
                seg.cropped_mask,
                seg.confidence,
                crop_region,
                bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        return (th, tw), new_segs

    def render_prompt(
        self, template, seg, index, total_segments, image_tensor, crop_region
    ):
        """Render Jinja2 template with seg context relative to cropped image"""
        # Get cropped image dimensions
        height, width = image_tensor.shape[0], image_tensor.shape[1]

        # Calculate bbox relative to the cropped region
        # bbox is in original image coords, crop_region defines the crop
        x1_crop, y1_crop, x2_crop, y2_crop = crop_region
        bx1, by1, bx2, by2 = seg.bbox

        # Convert to crop-relative coordinates
        bbox_rel = (
            bx1 - x1_crop,
            by1 - y1_crop,
            bx2 - x1_crop,
            by2 - y1_crop,
        )

        # Normalized bbox (0.0 to 1.0)
        bbox_norm = (
            bbox_rel[0] / width if width > 0 else 0,
            bbox_rel[1] / height if height > 0 else 0,
            bbox_rel[2] / width if width > 0 else 0,
            bbox_rel[3] / height if height > 0 else 0,
        )

        context = {
            "label": seg.label if seg.label else "",
            "index": index,
            "total_segments": total_segments,
            "confidence": seg.confidence,
            "width": width,
            "height": height,
            "bbox": bbox_rel,
            "bbox_norm": bbox_norm,
            "has_label": bool(seg.label),
        }

        # Check if template contains Jinja2 syntax
        if not re.search(r"\{[{%#]", template):
            # No Jinja2 syntax, return as-is (backward compatible)
            return template

        # Render with Jinja2
        try:
            env = jinja2.Environment(autoescape=False)
            tpl = env.from_string(template)
            return tpl.render(**context)
        except jinja2.exceptions.TemplateError as e:
            # Fail the node on template errors
            raise ValueError(
                f"Template rendering failed for segment {index}: {str(e)}. "
                f"Available variables: {', '.join(sorted(context.keys()))}"
            )

    def caption_segs(
        self,
        segs,
        prompt_template,
        model,
        projector,
        seed,
        temperature,
        max_tokens,
        n_ctx,
        n_gpu_layers,
        unload_model,
        fallback_image_opt=None,
    ):
        shape, seg_list = segs

        if len(seg_list) == 0:
            logging.warning("LlamacppSEGSCaptionerV2: Empty SEGS list provided")
            return ([],)

        # Validate model
        if model == "(no models found)":
            return (["(error: no models found)"],)

        model_path = folder_paths.get_full_path("LLM", model)
        if not model_path or not os.path.exists(model_path):
            return ([f"(error: model file not found: {model})"],)

        # Resolve projector path (required for image captioning)
        projector_path = folder_paths.get_full_path("LLM", projector)
        if not projector_path or not os.path.exists(projector_path):
            return (
                [
                    f"(error: Projector model not found: {projector}. Vision projector required for image captioning.)"
                ],
            )

        if fallback_image_opt is not None:
            segs = self.segs_scale_match(segs, fallback_image_opt.shape)
            shape, seg_list = segs

        captions = []

        logging.info(
            f"LlamacppSEGSCaptionerV2: Processing {len(seg_list)} segments with model {model}"
        )

        # Load model once for all segments
        clear_memory_for_llamacpp()
        cache_key = f"{model_path}_{n_ctx}_{n_gpu_layers}_{projector_path}"

        try:
            llm_model = get_or_load_model(
                model_path, n_ctx, n_gpu_layers, projector_path, has_images=True
            )
        except Exception as e:
            logging.error(f"LlamacppSEGSCaptionerV2: Failed to load model: {e}")
            return ([f"(error: failed to load model: {str(e)})"],)

        for idx, seg in enumerate(seg_list, start=1):
            image_tensor = None

            if seg.cropped_image is not None:
                image_tensor = seg.cropped_image
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]
            elif fallback_image_opt is not None:
                ref_image = (
                    fallback_image_opt[0].unsqueeze(0)
                    if len(fallback_image_opt.shape) == 4
                    else fallback_image_opt.unsqueeze(0)
                )
                x1, y1, x2, y2 = seg.crop_region
                cropped_image = ref_image[:, y1:y2, x1:x2, :]
                image_tensor = cropped_image[0]
                logging.info(
                    f"LlamacppSEGSCaptionerV2: Seg {idx} cropped from fallback image using crop_region {seg.crop_region}"
                )
            else:
                logging.warning(
                    f"LlamacppSEGSCaptionerV2: Seg {idx} has no cropped_image and no fallback provided"
                )
                captions.append("(no image)")
                continue

            # Render prompt with seg context
            try:
                rendered_prompt = self.render_prompt(
                    prompt_template,
                    seg,
                    idx,
                    len(seg_list),
                    image_tensor,
                    seg.crop_region,
                )
            except ValueError as e:
                logging.error(f"LlamacppSEGSCaptionerV2: {e}")
                raise

            try:
                print(f"\n\033[36m[Seg {idx}/{len(seg_list)}]\033[0m", flush=True)
                print(f"\033[33mPrompt: {rendered_prompt}\033[0m", flush=True)
                print()

                # Build messages with image
                images_batch = image_tensor.unsqueeze(0)  # [1, H, W, C]
                content = [{"type": "text", "text": rendered_prompt}]
                for img in images_batch:
                    img_data_url = convert_image_to_data_url(img)
                    content.append(
                        {"type": "image_url", "image_url": {"url": img_data_url}}
                    )
                messages = [{"role": "user", "content": content}]

                # Generate with streaming
                stream = llm_model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed + idx - 1 if seed > 0 else None,
                    stream=True,
                )

                caption = ""
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        caption += delta
                        print(f"\033[32m{delta}\033[0m", end="", flush=True)

                print()  # Newline after completion

                # Get timing statistics from llama.cpp
                timings = llama_cpp.llama_perf_context(llm_model._ctx.ctx)
                prompt_tokens = timings.n_p_eval
                completion_tokens = timings.n_eval
                total_time_s = (timings.t_p_eval_ms + timings.t_eval_ms) / 1000.0

                if completion_tokens > 0 and timings.t_eval_ms > 0:
                    tokens_per_sec = completion_tokens / (timings.t_eval_ms / 1000.0)
                    logging.info(
                        f"LlamacppSEGSCaptionerV2 seg {idx}: prompt {prompt_tokens} tokens - response {completion_tokens} tokens in {total_time_s:.2f}s ({tokens_per_sec:.2f} tokens/s)"
                    )

                caption = caption.strip().replace("\n", " ").replace("\r", " ")
                captions.append(caption)

            except torch.cuda.OutOfMemoryError:
                error = "CUDA OOM. Try reducing n_gpu_layers or use smaller model."
                logging.error(f"LlamacppSEGSCaptionerV2: {error}")
                captions.append(f"({error})")
            except Exception as e:
                logging.error(
                    f"LlamacppSEGSCaptionerV2: Failed to caption seg {idx}: {e}"
                )
                captions.append("(captioning failed)")

        logging.info(
            f"LlamacppSEGSCaptionerV2: Completed captioning {len(seg_list)} segments"
        )

        # Unload model if requested
        if unload_model:
            unload_model_from_cache(cache_key)
            clear_memory_for_llamacpp()

        return (captions,)


NODE_CLASS_MAPPINGS = {
    "yaSEGSCaptioner": SEGSCaptioner,
    "yaSEGSCaptionerV2": SEGSCaptionerV2,
    "yaLlamacppSEGSCaptioner": LlamacppSEGSCaptioner,
    "yaLlamacppSEGSCaptionerV2": LlamacppSEGSCaptionerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaSEGSCaptioner": "SEGS Captioner",
    "yaSEGSCaptionerV2": "SEGS Captioner V2",
    "yaLlamacppSEGSCaptioner": "Llamacpp SEGS Captioner",
    "yaLlamacppSEGSCaptionerV2": "Llamacpp SEGS Captioner V2",
}
