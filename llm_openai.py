import base64
import io
import logging
import re
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import requests
import tenacity
import torch
from openai import OpenAI
from PIL import Image, ImageOps

# Set the logging level for httpx and httpcore to WARNING or ERROR
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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
            content.append({
                "type": "text",
                "text": prompt,
            })

            # Convert first image to base64 (similar to LLMChat)
            img_tensor = images[0]
            img_data = 255.0 * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                },
            })

            messages.append({
                "role": "user",
                "content": content,
            })
        else:
            messages.append({"role": "user", "content": prompt})

        print(f"OpenAI API Request to {base_url}:")
        print(f"Model: {model_name}")
        print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed if seed > 0 else None,
            )

            result = response.choices[0].message.content
            print(f"OpenAI Response: {result[:200]}{'...' if len(result) > 200 else ''}")

            return (result.strip(),)

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
                "ratio": (["1:1", "2:3", "3:2"], {"default": "1:1"}),
                "num_images": (["1", "2", "4"], {"default": "4"}),
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

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, max=10),
        stop=tenacity.stop_after_attempt(5),
    )
    def download_image(self, url):
        response = requests.get(url)
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
        ratio,
        num_images,
        seed,
        images=None,
    ):
        client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )

        full_prompt = (
            f"{prompt}\n\nOutput image ratio: {ratio}\nOutput image count: {num_images}"
        )
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": full_prompt,
            }
        ]
        if images is not None:
            for img in self.encode_images_to_base64(images):
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img}",
                        },
                    }
                )

        print("Generating image, please wait...")
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

        resp_content = ""
        # handle stream output to console
        for chunk in resp_stream:
            content = getattr(chunk.choices[0].delta, "content", None)
            print(content, end="", flush=True)
            resp_content += content if content else ""
        print("\nImage generation completed.")

        # extract the image links from the response
        pattern = r"\[([^\]]+)\]\((https?://filesystem\.site[^\)]+)\)"
        matches = re.findall(pattern, resp_content)
        result_links = []
        for match in matches:
            link = match[1]
            if "/download/" not in link:
                result_links.append(link)

        # download those images and convert them to PIL images
        images = []
        for link in result_links:
            images.append(self.download_image(link))

        # convert the images to tensors
        image_tensors = [self.image_to_tensor(img) for img in images]

        return (image_tensors, resp_content)


NODE_CLASS_MAPPINGS = {
    "yaOpenAIGenerate": OpenAIGenerate,
    "yaGPTImageGeneratorChat": GPTImageGeneratorChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaOpenAIGenerate": "OpenAI Generate",
    "yaGPTImageGeneratorChat": "GPT Image Generator Chat",
}