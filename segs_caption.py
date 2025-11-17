import base64
import gc
import logging
import time
from io import BytesIO

import numpy as np
import ollama
import torch
from PIL import Image

import comfy.model_management


def clear_memory_for_ollama():
    """
    Aggressively clear ComfyUI and PyTorch memory before invoking Ollama.
    This is useful when Ollama runs on the same machine and may need GPU memory.
    """
    # Unload all ComfyUI models
    comfy.model_management.unload_all_models()

    # Force Python garbage collection
    gc.collect()

    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Wait for everything to settle
    time.sleep(0.5)


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

        # Options for Ollama
        options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

        captions = []

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

            # Clear memory before Ollama call
            clear_memory_for_ollama()

            # Call Ollama to caption the image
            try:
                logging.info(f"SEGSCaptioner: Captioning seg {idx}/{len(seg_list)}...")

                options["seed"] = seed + idx - 1  # Increment seed per seg

                response = client.generate(
                    model=model,
                    prompt=prompt_template,
                    images=[img_b64],
                    options=options,
                    keep_alive=f"{keep_alive}s",
                )

                caption = response["response"].strip()

                # Remove any newlines from caption to keep format clean
                caption = caption.replace("\n", " ").replace("\r", " ")

                captions.append(f"[{label}] {caption}")

                logging.info(
                    f"SEGSCaptioner: Seg {idx} captioned: {caption[:50]}{'...' if len(caption) > 50 else ''}"
                )

            except Exception as e:
                logging.error(f"SEGSCaptioner: Failed to caption seg {idx}: {e}")
                captions.append(f"[{label}] (captioning failed)")

        # Format output as [LAB] wildcard format
        wildcard_output = "[LAB]\n" + "\n".join(captions)

        logging.info(f"SEGSCaptioner: Completed captioning {len(seg_list)} segments")

        return (wildcard_output,)


NODE_CLASS_MAPPINGS = {
    "yaSEGSCaptioner": SEGSCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaSEGSCaptioner": "SEGS Captioner",
}
