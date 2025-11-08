import logging
import time
from typing import Tuple

import ollama
from aiohttp import web
from ollama import Client

import comfy.model_management
from server import PromptServer


# Set the logging level for httpx and httpcore to WARNING or ERROR
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
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
                "thinking": (["ON", "OFF", "HIGH", "MEDIUM", "LOW"], {"default": "OFF"}),
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
        full_reasoning = ""
        print("Starting Ollama generation (streaming):")

        if thinking == "OFF":
            thinking = False
        elif thinking == "ON":
            thinking = True
        elif thinking == "HIGH":
            thinking = "high"
        elif thinking == "MEDIUM":
            thinking = "medium"
        elif thinking == "LOW":
            thinking = "low"

        stream = client.generate(
            model=model,
            system="You are a willing AI assistant. You will follow user's instructions exactly.",
            prompt=prompt,
            options=options,
            keep_alive=f"{keep_alive}m",
            format=format,
            think=thinking,  # type: ignore
            stream=True,
        )

        # Process the stream and log to console
        for chunk in stream:
            if "thinking" in chunk:
                think_text = chunk["thinking"]
                print(f"\033[33m{think_text}\033[0m", end="", flush=True)
                full_reasoning += think_text
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
            # remove thinking phase: content between </think> and </think> tags
            think_start = full_response.find("</think>")
            if think_start != -1:
                think_end = full_response.find("</think>", think_start)
                if think_end != -1:
                    full_response = (
                        full_response[:think_start]
                        + full_response[think_end + len("</think>") :]
                    ).strip()
        elif full_reasoning:
            full_response = f"</think>{full_reasoning}</think>\n{full_response}"

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
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
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

        # remove thinking phase: content between </think> and </think> tags
        think_start = phase1_msg_content.find("</think>")
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
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
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

        # remove thinking phase: content between </think> and </think> tags
        think_start = phase1_msg_content.find("</think>")
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
            think_start = phase2_msg_content.find("</think>")
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


NODE_CLASS_MAPPINGS = {
    "yaOllamaGenerate": OllamaGenerate,
    "yaOllamaChat": OllamaChat,
    "yaOllamaChatDual": OllamaChatDual,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaOllamaGenerate": "Ollama Generate",
    "yaOllamaChat": "Ollama Chat",
    "yaOllamaChatDual": "Ollama Chat Dual Round",
}