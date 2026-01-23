import base64
import io
import logging
import re
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import requests
import torch
from openai import OpenAI
from PIL import Image, ImageOps

# Set the logging level for httpx and httpcore to WARNING or ERROR
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Create logger for this module
logger = logging.getLogger(__name__)


class OpenAIGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "What is Art?"}),
                "model_name": (
                    "STRING",
                    {"default": "gpt-4o-mini"},
                ),
                "base_url": (
                    "STRING",
                    {"default": "https://api.openai.com/v1/"},
                ),
                "api_key": (
                    "STRING",
                    {"default": "sk-XXXXX"},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 2, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 8192, "step": 1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 1, "step": 0.05},
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0, "min": -2, "max": 2, "step": 0.1},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0, "min": -2, "max": 2, "step": 0.1},
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional system prompt",
                    },
                ),
                "images": ("IMAGE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "openai_generate"
    CATEGORY = "Yet Another LLM"

    def openai_generate(
        self,
        prompt,
        model_name,
        base_url,
        api_key,
        seed,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        system_prompt="",
        images=None,
    ):
        _ = seed  # For ComfyUI re-run functionality

        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = []

        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        # Handle images if provided
        if images is not None and len(images) > 0:
            content = []
            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            # Convert first image to base64 (similar to LLMChat)
            img_tensor = images[0]
            img_data = 255.0 * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                    },
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        print(f"OpenAI API Request to {base_url}:")
        print(f"Model: {model_name}")
        print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        try:
            # Use streaming API
            response_stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed if seed > 0 else None,
                stream=True,
            )

            # Process the stream and accumulate the response
            full_response = ""
            print("OpenAI Response (streaming): ", end="", flush=True)

            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(f"\033[32m{content}\033[0m", end="", flush=True)
                    full_response += content

            print()  # New line after completion

            return (full_response.strip(),)

        except Exception as e:
            error_msg = f"OpenAI API Error: {str(e)}"
            print(error_msg)
            return (error_msg,)


class GPTImageGeneratorChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (
                    "STRING",
                    {"default": "gpt-4o-image-vip", "multiline": False},
                ),
                "api_url": (
                    "STRING",
                    {"default": "https://api.tu-zi.com/v1", "multiline": False},
                ),
                "api_key": ("STRING", {"multiline": False}),
                "api_mode": (
                    ["chat", "generations"],
                    {"default": "chat"},
                ),
                "size": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "5:4", "4:5", "16:9", "9:16"],
                    {"default": "1:1"},
                ),
                "num_images": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 66666666, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "API Respond")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate_image"
    CATEGORY = "Yet Another LLM"

    def encode_images_to_base64(self, image_tensor, max_dimension=1024, quality=85):
        base64_images = []
        batch_size = image_tensor.shape[0]
        for i in range(batch_size):
            input_image = image_tensor[i].cpu().numpy()
            input_image = (input_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(input_image)
            original_width, original_height = pil_image.width, pil_image.height
            if original_width > max_dimension or original_height > max_dimension:
                if original_width > original_height:
                    new_width = max_dimension
                    new_height = int(original_height * (max_dimension / original_width))
                else:
                    new_height = max_dimension
                    new_width = int(original_width * (max_dimension / original_height))
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)
        return base64_images

    def download_image(self, url):
        # No timeout - allow unlimited time for download, no retries
        response = requests.get(url, timeout=None)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    def image_to_tensor(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image)[None,]
        return image_tensor

    def generate_image(
        self,
        prompt,
        model,
        api_url,
        api_key,
        api_mode,
        size,
        num_images,
        seed,
        images=None,
    ):
        logger.info("=" * 80)
        logger.info("GPTImageGeneratorChat - Starting image generation")
        logger.info("=" * 80)
        logger.info(f"API URL: {api_url}")
        logger.info(f"API Mode: {api_mode}")
        logger.info(f"Model: {model}")
        logger.info(f"Size: {size}")
        logger.info(f"Number of images: {num_images}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(
            f"Prompt preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
        )

        # Handle images as either a single tensor or a list from ImageLister
        image_list = []
        if images is not None:
            if isinstance(images, list):
                # From ImageLister - already a list
                image_list = images
                logger.info(f"Images provided: Yes ({len(images)} images from list)")
            else:
                # Single IMAGE input - wrap in list
                image_list = [images]
                logger.info("Images provided: Yes (1 image)")
        else:
            logger.info("Images provided: No")

        logger.info("-" * 80)

        # Create OpenAI client with no timeout and no retries
        client = OpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=None,  # No timeout
            max_retries=0,  # No retries - generation is slow
        )

        # Use generations endpoint (DALL-E format)
        if api_mode == "generations":
            logger.info("Using /images/generations endpoint (DALL-E format)")

            # Build prompt with image reference if provided
            full_prompt = prompt
            if image_list:
                # For generations API, we can include image URLs in the prompt
                # Note: The API expects image URLs in the prompt text
                # Image-to-image may require chat mode
                logger.warning(
                    f"Image input provided ({len(image_list)} images). Consider using 'chat' mode for image-to-image generation."
                )
                logger.warning(
                    "The 'generations' endpoint may not support image inputs properly."
                )

            try:
                logger.info(f"Sending request to: {api_url}/images/generations")
                logger.info("Request parameters:")
                logger.info(f"  - model: {model}")
                logger.info(f"  - prompt: {full_prompt[:100]}...")
                logger.info(f"  - n: {num_images}")
                logger.info(f"  - size: {size}")
                logger.info("  - response_format: url")

                # Call the images.generate API (always use url format)
                response = client.images.generate(
                    model=model,
                    prompt=full_prompt,
                    n=num_images,
                    size=size,
                    response_format="url",
                )

                logger.info("Response received successfully")
            except Exception as e:
                error_msg = (
                    f"ERROR in images.generate API call: {type(e).__name__}: {str(e)}"
                )
                logger.error(error_msg)
                import traceback

                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                return ([], error_msg)

            resp_content = f"Generated {len(response.data) if response.data else 0} images using model {model}\n"
            logger.info(
                f"API returned {len(response.data) if response.data else 0} image(s)"
            )

            # Download images from URLs
            image_tensors = []
            if response.data:
                logger.info(f"Processing {len(response.data)} URL response(s)...")
                for idx, img_data in enumerate(response.data):
                    if img_data.url:
                        url = img_data.url
                        logger.info(f"  Image {idx + 1}/{len(response.data)}: {url}")
                        resp_content += f"Image {idx + 1}: {url}\n"
                        try:
                            logger.info(f"    Downloading image {idx + 1}...")
                            pil_image = self.download_image(url)
                            logger.info(
                                f"    Image {idx + 1} downloaded: {pil_image.size[0]}x{pil_image.size[1]} {pil_image.mode}"
                            )
                            image_tensors.append(self.image_to_tensor(pil_image))
                            logger.info(
                                f"    Image {idx + 1} converted to tensor successfully"
                            )
                        except Exception as e:
                            error_msg = f"ERROR downloading/processing image {idx + 1}: {type(e).__name__}: {str(e)}"
                            logger.error(error_msg)
                            resp_content += f"  ERROR: {error_msg}\n"
                    else:
                        logger.warning(f"  Image {idx + 1} has no URL")
            else:
                logger.warning("No image data in response")

            logger.info(
                f"Image generation completed. Successfully processed {len(image_tensors)} image(s)."
            )
            logger.info("=" * 80)
            return (image_tensors, resp_content)

        # Use chat completions endpoint (original behavior)
        else:
            logger.info("Using chat completions endpoint (streaming)")
            full_prompt = f"{prompt}\n\nOutput image ratio: {size}\nOutput image count: {num_images}"
            user_content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": full_prompt,
                }
            ]
            if image_list:
                logger.info(f"Processing {len(image_list)} input image(s)...")
                # image_list is a list of tensors (each can have different resolutions)
                for img_idx, img_tensor in enumerate(image_list):
                    logger.info(f"  Image {img_idx + 1}: shape {img_tensor.shape}")
                    # Encode each image tensor
                    base64_images = self.encode_images_to_base64(img_tensor)
                    for b64_idx, img_b64 in enumerate(base64_images):
                        logger.info(
                            f"    Adding sub-image {b64_idx + 1} to request (base64 length: {len(img_b64)} chars)"
                        )
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                },
                            }
                        )

            logger.info(f"Sending streaming request to: {api_url}/chat/completions")
            logger.info(
                f"Message content parts: {len(user_content)} (text + {len(user_content) - 1} images)"
            )

            try:
                resp_stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": user_content,
                        }  # type: ignore
                    ],
                    seed=seed,
                    stream=True,
                )

                logger.info("Stream connection established. Receiving response:")
                logger.info("-" * 80)
                resp_content = ""
                chunk_count = 0
                # handle stream output to console
                for chunk in resp_stream:
                    chunk_count += 1
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        print(content, end="", flush=True)
                        resp_content += content

                print()
                logger.info("-" * 80)
                logger.info(
                    f"Stream completed. Received {chunk_count} chunks, total {len(resp_content)} characters"
                )

            except Exception as e:
                error_msg = (
                    f"ERROR in chat.completions API call: {type(e).__name__}: {str(e)}"
                )
                logger.error(error_msg)
                import traceback

                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                logger.info("=" * 80)
                return ([], error_msg)

            # extract the image links from the response
            logger.info("Extracting image URLs from response...")

            # Try both markdown image format ![alt](url) and link format [text](url)
            # Match any http/https URL, not just specific domains
            image_pattern = r"!\[([^\]]*)\]\((https?://[^\)]+)\)"
            link_pattern = r"\[([^\]]+)\]\((https?://[^\)]+)\)"

            # First try markdown images
            matches = re.findall(image_pattern, resp_content)
            if matches:
                logger.info(
                    f"Found {len(matches)} markdown image(s) with ![...](url) format"
                )
            else:
                # Fall back to markdown links
                matches = re.findall(link_pattern, resp_content)
                logger.info(
                    f"Found {len(matches)} markdown link(s) with [text](url) format"
                )

            result_links = []
            for idx, match in enumerate(matches):
                link = match[1]
                logger.info(f"  Match {idx + 1}: '{match[0]}' -> {link}")
                result_links.append(link)

            logger.info(f"Total images to download: {len(result_links)}")

            # download those images and convert them to PIL images
            result_images = []
            for idx, link in enumerate(result_links):
                try:
                    logger.info(
                        f"  Downloading image {idx + 1}/{len(result_links)}: {link}"
                    )
                    pil_image = self.download_image(link)
                    logger.info(
                        f"    Downloaded: {pil_image.size[0]}x{pil_image.size[1]} {pil_image.mode}"
                    )
                    result_images.append(pil_image)
                except Exception as e:
                    error_msg = f"ERROR downloading image {idx + 1}: {type(e).__name__}: {str(e)}"
                    logger.error(error_msg)

            # convert the images to tensors
            logger.info(f"Converting {len(result_images)} image(s) to tensors...")
            image_tensors = [self.image_to_tensor(img) for img in result_images]
            logger.info(
                f"Successfully converted {len(image_tensors)} image(s) to tensors"
            )

            logger.info("=" * 80)
            return (image_tensors, resp_content)


class TuZiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (
                    "STRING",
                    {"default": "gemini-3-pro-image-preview-2k", "multiline": False},
                ),
                "api_url": (
                    "STRING",
                    {"default": "https://api.tu-zi.com/v1", "multiline": False},
                ),
                "api_key": ("STRING", {"multiline": False}),
                "size": (
                    "STRING",
                    {
                        "default": "2K",
                        "tooltip": "Available size options: 1K, 2K, 4K, 1024x1024, 2048x2048, 1024x1536, 1536x1024, 1536x2048, 2048x1536, 2048x3072, 3072x2048",
                    },
                ),
                "num_images": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "response")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate_image"
    CATEGORY = "Yet Another LLM"

    def encode_images_to_base64(self, image_tensor, max_dimension=1024, quality=85):
        """Encode image tensors to base64 format"""
        base64_images = []
        batch_size = image_tensor.shape[0]
        for i in range(batch_size):
            input_image = image_tensor[i].cpu().numpy()
            input_image = (input_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(input_image)

            # Resize if needed
            original_width, original_height = pil_image.width, pil_image.height
            if original_width > max_dimension or original_height > max_dimension:
                if original_width > original_height:
                    new_width = max_dimension
                    new_height = int(original_height * (max_dimension / original_width))
                else:
                    new_height = max_dimension
                    new_width = int(original_width * (max_dimension / original_height))
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)
        return base64_images

    def download_image(self, url):
        """Download image from URL with no timeout"""
        response = requests.get(url, timeout=None)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    def image_to_tensor(self, image):
        """Convert PIL image to tensor"""
        image = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image)[None,]
        return image_tensor

    def generate_image(
        self,
        prompt,
        model,
        api_url,
        api_key,
        size,
        num_images,
        seed,
        images=None,
    ):
        # Check if this is a seedream model
        is_seedream = "seedream" in model.lower()

        logger.info("=" * 80)
        logger.info("TuZiImageGenerator - Starting image generation")
        logger.info("=" * 80)
        logger.info(f"API URL: {api_url}")
        logger.info(f"Model: {model}")
        logger.info(f"Size: {size}")
        if is_seedream:
            logger.info("Watermark: False (seedream model detected)")
        logger.info(f"Number of images: {num_images}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(
            f"Prompt preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
        )

        # Handle input images
        image_data = None
        if images is not None:
            if isinstance(images, list):
                image_list = images
                logger.info(f"Input images: {len(images)} images from list")
            else:
                image_list = [images]
                logger.info("Input images: 1 image")

            # Encode all images to base64
            all_base64 = []
            for img_idx, img_tensor in enumerate(image_list):
                logger.info(f"  Encoding image {img_idx + 1}: shape {img_tensor.shape}")
                base64_images = self.encode_images_to_base64(img_tensor)
                all_base64.extend(base64_images)

            # Format as data URLs
            if len(all_base64) == 1:
                image_data = f"data:image/jpeg;base64,{all_base64[0]}"
            else:
                image_data = [f"data:image/jpeg;base64,{b64}" for b64 in all_base64]

            logger.info(f"  Total encoded images: {len(all_base64)}")
        else:
            logger.info("Input images: None")

        logger.info("-" * 80)

        # Build request parameters
        params = {
            "model": model,
            "prompt": prompt,
            "n": num_images,
            "size": size,
            "response_format": "url",
        }

        # Add watermark parameter only for seedream models
        if is_seedream:
            params["watermark"] = False

        # Add seed if provided (API supports -1 to 2147483647)
        if seed > 0:
            params["seed"] = seed

        # Add image data if provided
        if image_data is not None:
            params["image"] = image_data

        logger.info("Sending request to /images/generations endpoint")
        logger.info("Request parameters:")
        for key, value in params.items():
            if key == "image":
                if isinstance(value, list):
                    logger.info(f"  - {key}: [{len(value)} images (data URLs)]")
                else:
                    logger.info(f"  - {key}: [1 image (data URL)]")
            elif key == "prompt":
                logger.info(f"  - {key}: {value[:100]}...")
            else:
                logger.info(f"  - {key}: {value}")

        try:
            # Use raw HTTP request for custom API that supports image parameter
            url = f"{api_url.rstrip('/')}/images/generations"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(url, headers=headers, json=params, timeout=None)
            response.raise_for_status()
            response_data = response.json()

            logger.info("Response received successfully")

            # Parse response to match OpenAI format
            class ImageData:
                def __init__(self, url):
                    self.url = url

            class ImageResponse:
                def __init__(self, data):
                    self.data = [ImageData(item.get("url")) for item in data]

            response = ImageResponse(response_data.get("data", []))

        except Exception as e:
            error_msg = (
                f"ERROR in images.generate API call: {type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            import traceback

            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            logger.info("=" * 80)
            return ([], error_msg)

        # Process response
        resp_content = f"Generated {len(response.data) if response.data else 0} images using model {model}\n"
        logger.info(
            f"API returned {len(response.data) if response.data else 0} image(s)"
        )

        # Download images from URLs
        image_tensors = []
        if response.data:
            logger.info(f"Processing {len(response.data)} URL response(s)...")
            for idx, img_data in enumerate(response.data):
                if img_data.url:
                    url = img_data.url
                    logger.info(f"  Image {idx + 1}/{len(response.data)}: {url}")
                    resp_content += f"Image {idx + 1}: {url}\n"
                    try:
                        logger.info(f"    Downloading image {idx + 1}...")
                        pil_image = self.download_image(url)
                        logger.info(
                            f"    Downloaded: {pil_image.size[0]}x{pil_image.size[1]} {pil_image.mode}"
                        )
                        image_tensors.append(self.image_to_tensor(pil_image))
                        logger.info(
                            f"    Image {idx + 1} converted to tensor successfully"
                        )
                    except Exception as e:
                        error_msg = f"ERROR downloading/processing image {idx + 1}: {type(e).__name__}: {str(e)}"
                        logger.error(error_msg)
                        resp_content += f"  ERROR: {error_msg}\n"
                else:
                    logger.warning(f"  Image {idx + 1} has no URL")
        else:
            logger.warning("No image data in response")

        logger.info(
            f"Image generation completed. Successfully processed {len(image_tensors)} image(s)."
        )
        logger.info("=" * 80)
        return (image_tensors, resp_content)


NODE_CLASS_MAPPINGS = {
    "yaOpenAIGenerate": OpenAIGenerate,
    "yaGPTImageGeneratorChat": GPTImageGeneratorChat,
    "yaTuZiImageGenerator": TuZiImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaOpenAIGenerate": "OpenAI Generate",
    "yaGPTImageGeneratorChat": "GPT Image Generator Chat",
    "yaTuZiImageGenerator": "TuZi Image Generator",
}
