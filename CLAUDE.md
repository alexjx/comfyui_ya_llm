# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom node plugin that provides LLM integration capabilities. It supports both OpenAI-compatible APIs and Ollama for generating and editing prompts within ComfyUI workflows.

## Architecture

### Node System

The plugin follows ComfyUI's custom node architecture:

- **`__init__.py`**: Entry point that exports `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, and `WEB_DIRECTORY`
- **`chat.py`**: Contains all node class definitions and the main API endpoint
- **`web/js/yallm.js`**: Frontend extension for dynamic model loading in Ollama nodes and dynamic inputs for TextTemplate

### Custom Data Types

The plugin defines custom ComfyUI data types for chaining nodes:

- `LLM_API_MODEL`: Wrapper around OpenAI client instances
- `MSG_LIST`: List of message dictionaries for multi-turn conversations

### Node Categories

All nodes are registered under the category `"Yet Another LLM"` and include:

1. **API-based nodes**: `LLMApiModelLoader`, `LLMChat` (OpenAI-compatible APIs)
2. **Ollama nodes**: `OllamaGenerate`, `OllamaChat`, `OllamaChatDual`
3. **Utility nodes**: `TextTemplate`, `TextExtract`, `TextRemove`
4. **Image generation**: `GPTImageGeneratorChat`

### Key Implementation Details

**Message Chaining**: The `LLMChat` node accepts an optional `messages` parameter of type `MSG_LIST`, enabling multi-turn conversations by passing dialog history between nodes.

**Template System**: Uses Jinja2 for rendering user prompts with variables. The `TextTemplate` node supports dynamic inputs (text.py) with no limit, using **kwargs to accept any number of inputs. Inputs are rendered using Jinja2 template globals.

**Thinking Mode Handling**: Ollama nodes support thinking modes (HIGH/MEDIUM/LOW/ON/OFF) and include logic to strip `<think>...</think>` tags from responses unless `keep_reason` or `keep_thinking` is enabled (chat.py:516-528, 666-674, 845-856).

**Image Support**: `LLMChat` converts ComfyUI IMAGE tensors to base64-encoded PNG for vision model APIs (chat.py:179-205). Images are normalized from [0,1] to [0,255] uint8.

**Model Management**: Before Ollama generation, calls `comfy.model_management.unload_all_models()` to free GPU memory (chat.py:469, 644, 824).

**Dynamic Model Loading**: The frontend JavaScript (web/js/yallm.js) fetches available Ollama models via the `/yallm/get_ollama_models` endpoint and dynamically populates dropdown widgets.

**Dynamic Inputs**: The TextTemplate node uses JavaScript in web/js/yallm.js to provide unlimited dynamic inputs. Users can connect as many inputs as needed, which auto-renumber sequentially (input1, input2, input3, etc.).

## Development Commands

### Running in ComfyUI

This is a ComfyUI custom node. To use it:

```bash
# Navigate to ComfyUI's custom_nodes directory
cd /path/to/ComfyUI/custom_nodes/comfyui_ya_llm

# Install dependencies
pip install -r requirements.txt

# ComfyUI will automatically load this node on startup
```

### Testing

No formal test suite exists. Testing is done through ComfyUI's node graph interface.

## Dependencies

- `openai>=1.40.1`: OpenAI API client
- `ollama`: Ollama Python client
- `pillow`: Image processing
- `numpy`: Array operations
- `jinja2`: Template rendering
- `tenacity`: Retry logic for image downloads
- `torch`: PyTorch (provided by ComfyUI)
- `aiohttp`: Async HTTP server (provided by ComfyUI)

## API Endpoints

**`POST /yallm/get_ollama_models`**: Fetches available models from an Ollama instance. Takes `url` in request body and returns a list of model names (chat.py:30-41).

## Important Patterns

### Adding New Nodes

1. Create a new class with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, and `CATEGORY` class attributes
2. Implement the function specified in `FUNCTION`
3. Add to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in the appropriate file (llm_openai.py, llm_ollama.py, or text.py)
4. If the node requires frontend interaction, extend web/js/yallm.js

### Seed Handling

Seed parameters are used for ComfyUI's workflow caching and re-execution. The value is often assigned but not directly used (e.g., `_ = seed` in chat.py:164), allowing workflow re-runs to trigger when seed changes.

### Streaming Responses

Ollama nodes use streaming APIs for real-time output. The `OllamaGenerate` node prints color-coded output: yellow for thinking, green for response (chat.py:500-508).

## Configuration

API keys and endpoints are configured per-node in ComfyUI's interface. The `.env` file is not used for configuration but may contain developer environment variables.
