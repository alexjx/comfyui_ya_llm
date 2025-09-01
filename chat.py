import base64
import io
import json
import logging
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import jinja2
import numpy as np
import ollama
import requests
import tenacity
import torch
from aiohttp import web
from ollama import Client
from openai import OpenAI
from PIL import Image, ImageOps

import comfy.model_management
from server import PromptServer


@PromptServer.instance.routes.post("/yallm/get_ollama_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url")
    client = Client(host=url)
    models = client.list().get("models", [])
    try:
        models = [model["model"] for model in models]
        return web.json_response(models)
    except Exception as e:
        models = [model["name"] for model in models]
        return web.json_response(models)


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
                "seed": ("INT", {"default": 0, "min": 0}),
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
                "template": (
                    "STRING",
                    {"multiline": True, "default": "{{user_prompt}}"},
                ),
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
                "input1": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "template": (
                    "STRING",
                    {"multiline": True, "default": "{{input1}}"},
                ),
            },
            "optional": {
                "input2": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "input3": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "input4": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "input5": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "input6": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "input7": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
                "input8": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rendered_text",)
    CATEGORY = "Yet Another LLM"
    FUNCTION = "render"

    def render(
        self,
        input1: str,
        template: str,
        input2: str = "",
        input3: str = "",
        input4: str = "",
        input5: str = "",
        input6: str = "",
        input7: str = "",
        input8: str = "",
    ):
        template_env = jinja2.Environment(autoescape=False)
        template_str = template_env.from_string(template)
        if input1 is not None and input1 != "":
            input1 = input1.strip()
            template_str.globals["input1"] = input1
        if input2 is not None and input2 != "":
            input2 = input2.strip()
            template_str.globals["input2"] = input2
        if input3 is not None and input3 != "":
            input3 = input3.strip()
            template_str.globals["input3"] = input3
        if input4 is not None and input4 != "":
            input4 = input4.strip()
            template_str.globals["input4"] = input4
        if input5 is not None and input5 != "":
            input5 = input5.strip()
            template_str.globals["input5"] = input5
        if input6 is not None and input6 != "":
            input6 = input6.strip()
            template_str.globals["input6"] = input6
        if input7 is not None and input7 != "":
            input7 = input7.strip()
            template_str.globals["input7"] = input7
        if input8 is not None and input8 != "":
            input8 = input8.strip()
            template_str.globals["input8"] = input8
        rendered_text = template_str.render()
        return (rendered_text,)


class TextExtract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "begin": ("STRING", {}),
                "end": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_text",)
    CATEGORY = "Yet Another LLM"
    FUNCTION = "extract"

    def extract(
        self, input_text: str, begin: Optional[str] = None, end: Optional[str] = None
    ) -> Tuple[str]:
        if begin:
            start_idx = input_text.find(begin)
            if start_idx == -1:
                raise ValueError(f"Begin string '{begin}' not found in input text")
        else:
            start_idx = 0
        if end:
            end_idx = input_text.find(end, start_idx)
            if end_idx == -1:
                raise ValueError(f"End string '{end}' not found in input text")
        else:
            end_idx = len(input_text)
        return (input_text[start_idx:end_idx],)


class TextRemove:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"forceInput": True}),
                "begin": ("STRING", {}),
                "end": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    CATEGORY = "Yet Another LLM"
    FUNCTION = "remove"

    def remove(
        self, input_text: str, begin: Optional[str] = None, end: Optional[str] = None
    ) -> Tuple[str]:
        start_idx = input_text.find(begin)
        if start_idx == -1:
            return (input_text,)
        end_idx = input_text.find(end, start_idx)
        if end_idx == -1:
            return (input_text,)
        end_idx += len(end)
        result = input_text[:start_idx] + input_text[end_idx:]
        return (result.strip(),)


class OllamaGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "What is Art?"}),
                "url": (
                    "STRING",
                    {"multiline": False, "default": "http://127.0.0.1:11434"},
                ),
                "model": ((), {}),
                "seed": ("INT", {"default": 0, "min": 0, "step": 1}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 2, "step": 0.05},
                ),
                "num_ctx": (
                    "INT",
                    {"default": 2048, "min": 1, "max": 8192, "step": 1},
                ),
                "num_predict": (
                    "INT",
                    {"default": -1, "min": -2, "max": 4096, "step": 1},
                ),
                "keep_alive": ("INT", {"default": 0, "min": -1, "max": 60, "step": 1}),
                "thinking": ("BOOLEAN", {"default": False}),
                "keep_reason": ("BOOLEAN", {"default": False}),
                "format": (["text", "json", ""],),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ollama_generate"
    CATEGORY = "Yet Another LLM"

    def ollama_generate(
        self,
        prompt,
        url,
        model,
        seed,
        temperature,
        num_ctx,
        num_predict,
        keep_alive,
        keep_reason,
        thinking,
        format,
    ):
        client = ollama.Client(host=url)

        if format == "text":
            format = ""

        options = {
            "seed": seed,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

        model = model.strip()

        # unload models before proceed
        comfy.model_management.unload_all_models()
        time.sleep(1)  # wait for the model to be unloaded

        # Use streaming API
        full_response = ""
        print("Starting Ollama generation (streaming):")

        stream = client.generate(
            model=model,
            system="You are an willingly AI insistant. You will follow user's instructions exactly.",
            prompt=prompt,
            options=options,
            keep_alive=f"{keep_alive}m",
            format=format,
            think=thinking,
            stream=True,
        )

        # Process the stream and log to console
        for chunk in stream:
            if "response" in chunk:
                response_text = chunk["response"]
                print(f"\033[32m{response_text}\033[0m", end="", flush=True)
                full_response += response_text
            if "done" in chunk and chunk["done"]:
                print()  # New line after completion
                logging.info(
                    f"Generated: prompt {chunk.get('prompt_eval_count', 0)} token - response {chunk.get('eval_count', 0)} tokens in {chunk.get('total_duration', 0) / (10**9):.2f}s ({chunk.get('eval_count', 0) / chunk.get('eval_duration', 1) * (10**9):.2f} tokens/s)"
                )
                break

        if not keep_reason:
            # remove thinking phase: content between <think> and </think> tags
            think_start = full_response.find("<think>")
            if think_start != -1:
                think_end = full_response.find("</think>", think_start)
                if think_end != -1:
                    full_response = (
                        full_response[:think_start]
                        + full_response[think_end + len("</think>") :]
                    ).strip()

        full_response = full_response.strip()
        assert len(full_response) > 0, "Response is empty"

        return (full_response,)


class OllamaChat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an art expert, gracefully describing your knowledge in art domain.",
                        "title": "system",
                    },
                ),
                "prompt1": ("STRING", {"multiline": True, "default": "What is Art?"}),
                "prompt2": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "This is the second prompt for round two, leave it empty if you don't want to use it.",
                    },
                ),
                "url": (
                    "STRING",
                    {"multiline": False, "default": "http://127.0.0.1:11434"},
                ),
                "model": ((), {}),
                "seed": ("INT", {"default": 0, "min": 0, "step": 1}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 2, "step": 0.05},
                ),
                "num_ctx": (
                    "INT",
                    {"default": 2048, "min": 1, "max": 8192, "step": 1},
                ),
                "num_predict": (
                    "INT",
                    {"default": -1, "min": -2, "max": 4096, "step": 1},
                ),
                "keep_alive": (
                    "INT",
                    {"default": 1, "min": -1, "max": 3600, "step": 1},
                ),
                "format": (["text", "json", ""],),
            },
            "optional": {},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "response1",
        "response2",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Yet Another LLM"

    def ollama_chat(
        self,
        system,
        prompt1,
        prompt2,
        url,
        model,
        seed,
        temperature,
        num_ctx,
        num_predict,
        keep_alive,
        format,
    ):
        client = ollama.Client(host=url)

        if format == "text":
            format = ""

        # num_keep: int
        # seed: int
        # num_predict: int
        # top_k: int
        # top_p: float
        # tfs_z: float
        # typical_p: float
        # repeat_last_n: int
        # temperature: float
        # repeat_penalty: float
        # presence_penalty: float
        # frequency_penalty: float
        # mirostat: int
        # mirostat_tau: float
        # mirostat_eta: float
        # penalize_newline: bool
        # stop: Sequence[str]

        options = {
            "seed": seed,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

        model = model.strip()

        # unload models before proceed
        comfy.model_management.unload_all_models()
        time.sleep(1)  # wait for the model to be unloaded

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt1},
        ]

        # first round
        phase1_resp = client.chat(
            model=model,
            messages=messages,
            options=options,
            keep_alive=f"{keep_alive}s",
            format=format,
        )
        logging.info(
            f"Phase 1: duration {phase1_resp['total_duration'] / (10**9):.2f}s rate {phase1_resp['eval_count'] / phase1_resp['eval_duration'] * (10**9):.2f} tokens/s"
        )
        phase1_msg = phase1_resp["message"]["content"]
        phase1_msg_content = phase1_msg

        # remove thinking phase: content between <think> and </think> tags
        think_start = phase1_msg_content.find("<think>")
        if think_start != -1:
            think_end = phase1_msg_content.find("</think>", think_start)
            if think_end != -1:
                phase1_msg_content = (
                    phase1_msg_content[:think_start]
                    + phase1_msg_content[think_end + len("</think>") :]
                )

        # second round if prompt2 is provided
        phase2_msg = ""
        if prompt2:
            messages.append({"role": "assistant", "content": phase1_msg_content})
            messages.append({"role": "user", "content": prompt2})
            options["seed"] = seed + 1
            phase2_resp = client.chat(
                model=model,
                messages=messages,
                options=options,
                keep_alive=f"{keep_alive}s",
                format=format,
            )
            logging.info(
                f"Phase 2: duration {phase2_resp['total_duration'] / (10**9):.2f}s rate {phase2_resp['eval_count'] / phase2_resp['eval_duration'] * (10**9):.2f} tokens/s"
            )
            phase2_msg = phase2_resp["message"]["content"]

        return (phase1_msg, phase2_msg)


class OllamaChatDual:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an art expert, gracefully describing your knowledge in art domain.",
                        "title": "system",
                    },
                ),
                "prompt1": ("STRING", {"multiline": True, "default": "What is Art?"}),
                "prompt2": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "This is the second prompt for round two, leave it empty if you don't want to use it.",
                    },
                ),
                "url": (
                    "STRING",
                    {"multiline": False, "default": "http://127.0.0.1:11434"},
                ),
                "model1": ((), {}),
                "model2": ((), {}),
                "seed": ("INT", {"default": 0, "min": 0, "step": 1}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 2, "step": 0.05},
                ),
                "num_ctx": (
                    "INT",
                    {"default": 2048, "min": 1, "max": 8192, "step": 1},
                ),
                "num_predict": (
                    "INT",
                    {"default": -1, "min": -2, "max": 4096, "step": 1},
                ),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "keep_alive": (
                    "INT",
                    {"default": 1, "min": -1, "max": 3600, "step": 1},
                ),
                "keep_thinking": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "format": (["text", "json", ""],),
            },
            "optional": {},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "response1",
        "response2",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Yet Another LLM"

    def ollama_chat(
        self,
        system,
        prompt1,
        prompt2,
        url,
        model1,
        model2,
        seed,
        top_k,
        top_p,
        temperature,
        num_ctx,
        num_predict,
        tfs_z,
        keep_alive,
        keep_thinking,
        format,
    ):
        client = ollama.Client(host=url)

        if format == "text":
            format = ""

        # num_keep: int
        # seed: int
        # num_predict: int
        # top_k: int
        # top_p: float
        # tfs_z: float
        # typical_p: float
        # repeat_last_n: int
        # temperature: float
        # repeat_penalty: float
        # presence_penalty: float
        # frequency_penalty: float
        # mirostat: int
        # mirostat_tau: float
        # mirostat_eta: float
        # penalize_newline: bool
        # stop: Sequence[str]

        options = {
            "seed": seed,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "tfs_z": tfs_z,
        }

        model1 = model1.strip()
        model2 = model2.strip()

        # unload models before proceed
        comfy.model_management.unload_all_models()
        time.sleep(1)  # wait for the model to be unloaded

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt1},
        ]

        # first round
        phase1_resp = client.chat(
            model=model1,
            messages=messages,
            options=options,
            keep_alive=f"{keep_alive}s",
            format=format,
        )
        logging.info(
            f"Round 1: prompt {phase1_resp['prompt_eval_count']} token - response {phase1_resp['eval_count']} tokens in {phase1_resp['total_duration'] / (10**9):.2f}s ({phase1_resp['eval_count'] / phase1_resp['eval_duration'] * (10**9):.2f} tokens/s)"
        )
        phase1_msg = phase1_resp["message"]["content"]
        phase1_msg_content = phase1_msg

        # remove thinking phase: content between <think> and </think> tags
        think_start = phase1_msg_content.find("<think>")
        if think_start != -1:
            think_end = phase1_msg_content.find("</think>", think_start)
            if think_end != -1:
                phase1_msg_content = (
                    phase1_msg_content[:think_start]
                    + phase1_msg_content[think_end + len("</think>") :]
                )
                phase1_msg_content = phase1_msg_content.strip()
        if not keep_thinking:
            phase1_msg = phase1_msg_content

        # second round if prompt2 is provided
        phase2_msg = ""
        if prompt2:
            messages.append({"role": "assistant", "content": phase1_msg_content})
            messages.append({"role": "user", "content": prompt2})
            options["seed"] = seed + 1
            phase2_resp = client.chat(
                model=model2,
                messages=messages,
                options=options,
                keep_alive=f"{keep_alive}s",
                format=format,
            )
            logging.info(
                f"Round 2: prompt {phase2_resp['prompt_eval_count']} token - response {phase2_resp['eval_count']} tokens in {phase2_resp['total_duration'] / (10**9):.2f}s ({phase2_resp['eval_count'] / phase2_resp['eval_duration'] * (10**9):.2f} tokens/s)"
            )
            phase2_msg = phase2_resp["message"]["content"]
        phase2_msg_content = phase2_msg
        if not keep_thinking:
            think_start = phase2_msg_content.find("<think>")
            if think_start != -1:
                think_end = phase2_msg_content.find("</think>", think_start)
                if think_end != -1:
                    phase2_msg_content = (
                        phase2_msg_content[:think_start]
                        + phase2_msg_content[think_end + len("</think>") :]
                    )
                    phase2_msg_content = phase2_msg_content.strip()
            phase2_msg = phase2_msg_content

        return (phase1_msg, phase2_msg)


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
                "seed": ("INT", {"default": 66666666, "min": 0}),
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


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "yaLLMApiModelLoader": LLMApiModelLoader,
    "yaLLMChat": LLMChat,
    "yaLLMTextTemplate": TextTemplate,
    "yaTextExtract": TextExtract,
    "yaTextRemove": TextRemove,
    "yaOllamaChat": OllamaChat,
    "yaOllamaGenerate": OllamaGenerate,
    "yaOllamaChatDual": OllamaChatDual,
    "yaGPTImageGeneratorChat": GPTImageGeneratorChat,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "yaLLMApiModelLoader": "Load API Model",
    "yaLLMChat": "API LLM Chat",
    "yaLLMTextTemplate": "Text Template",
    "yaTextExtract": "Text Extract",
    "yaTextRemove": "Text Remove",
    "yaOllamaChat": "Ollama Chat",
    "yaOllamaGenerate": "Ollama Generate",
    "yaOllamaChatDual": "Ollama Chat Dual Round",
    "yaGPTImageGeneratorChat": "GPT Image Generator Chat",
}
