from openai import OpenAI
from PIL import Image
import jinja2
from typing import Optional, List
import json
import jinja2
import numpy as np
import io
import base64


class OpenAIApiModel:
    def __init__(self, model_name: str, base_url: str, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def chat(self, messages: list, max_tokens: int, temperature: float):
        print(f"LLM REQ:  {json.dumps(messages, indent=2)}")
        resp = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        msg = resp.choices[0].message
        print(f"LLM RESP: {json.dumps(msg.content, indent=2)}")
        return {"role": "assistant", "content": msg.content}


class LLMApiModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {
                        "default": "gpt-4o-mini",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "https://api.openai.com/v1/",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "sk-XXXXX",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LLM_API_MODEL",)
    RETURN_NAMES = ("llm_api_model",)
    CATEGORY = "Yet Another LLM"
    FUNCTION = "load"

    def load(self, model_name: str, base_url: str, api_key: str):
        return (
            OpenAIApiModel(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            ),
        )


class LLMChat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_model": ("LLM_API_MODEL", {}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "max_tokens": (
                    "FLOAT",
                    {"default": 1920, "min": 256, "max": 128000, "step": 128},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "messages": ("MSG_LIST", {}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "This is the system prompt sent to LLM",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": 'This is the user input, it will be rendered by template before sending it to LLM, reference it by "{{user_prompt}}"',
                    },
                ),
                "images": ("IMAGE", {"forceInput": True}),
                "template": ("STRING", {"multiline": True, "default": "{{user_prompt}}"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "MSG_LIST")
    RETURN_NAMES = ("response", "dialog")
    CATEGORY = "Yet Another LLM"
    FUNCTION = "chat"

    def chat(
        self,
        api_model: OpenAIApiModel,
        temperature: float,
        max_tokens: int,
        seed: int,
        messages: Optional[List] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        images=None,
        template: str = "{user_prompt}",
        unique_id=None,
    ):
        _ = seed  # this is only for the sake of re-run
        if not messages:
            messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Apply the template if provided
        if template:
            template_env = jinja2.Environment(autoescape=False)
            template_str = template_env.from_string(template)
            user_prompt = template_str.render(user_prompt=user_prompt)

        # if we have images, we need to convert them to text
        # FIXME: we are currently only take the first image
        if images is not None and len(images) > 0:
            # prepare content
            content = []
            # prepare user prompt
            if user_prompt:
                content.append(
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
                )
            # prepare image
            img_tensor = images[0]
            img_data = 255.0 * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
            # save the image as png and encode it with base64
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
            # add the content to the messages
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        elif user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        response = api_model.chat(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        messages.append(response)
        return response["content"].strip(), messages


class TextTemplate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_input": ("STRING", {"forceInput": True, }),
                "template": ("STRING", {"multiline": True, "default": "{{user_input}}" }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rendered_text",)
    CATEGORY = "Text Processing"
    FUNCTION = "render"

    def render(self, user_input: str, template: str):
        template_env = jinja2.Environment(autoescape=False)
        template_str = template_env.from_string(template)
        rendered_text = template_str.render(user_input=user_input)
        return (rendered_text,)

