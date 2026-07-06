# Yet Another LLM (YA-LLM) - ComfyUI Custom Nodes

A comprehensive collection of LLM integration nodes for ComfyUI, providing simple yet powerful tools for prompt generation, editing, and image captioning within your workflows.

## Overview

This custom node pack provides seamless integration with both OpenAI-compatible APIs and Ollama for local LLM inference. It's designed to be simple yet tunable, filling the gap between overly complex and overly simple LLM nodes.

## Features

- **OpenAI-Compatible API Support**: Work with OpenAI, Azure OpenAI, and any OpenAI-compatible endpoints
- **Ollama Integration**: Run local models with full parameter control
- **Vision Model Support**: Process images with vision-capable models
- **Multi-Turn Conversations**: Chain nodes for complex dialog flows
- **Template System**: Dynamic Jinja2-based text templating with unlimited inputs
- **Thinking Mode Support**: Control reasoning output in compatible models
- **Memory Management**: Automatic GPU memory optimization for Ollama
- **Image Generation**: Generate images using GPT vision models
- **SEGS Captioning**: Automatically caption segmented regions with vision models

## Installation

1. Navigate to your ComfyUI custom_nodes directory:
```bash
cd /path/to/ComfyUI/custom_nodes/
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/comfyui_ya_llm.git
```

3. Install dependencies:
```bash
cd comfyui_ya_llm
pip install -r requirements.txt
```

4. Restart ComfyUI

## Node Reference

All nodes are available under the category **"Yet Another LLM"** in ComfyUI.

### OpenAI-Compatible API Nodes

#### OpenAI Generate

Generate text responses using OpenAI-compatible APIs with support for vision models.

**Inputs:**
- `prompt` (STRING, required): The user prompt or question
- `model_name` (STRING, required): Model identifier (e.g., "gpt-4o-mini", "gpt-4o")
- `base_url` (STRING, required): API endpoint URL (default: "https://api.openai.com/v1/")
- `api_key` (STRING, required): Your API key
- `seed` (INT, required): Random seed for reproducibility (0 for random)
- `temperature` (FLOAT, required): Sampling temperature (0-2, default: 0.7)
- `max_tokens` (INT, required): Maximum tokens to generate (1-8192, default: 1024)
- `top_p` (FLOAT, required): Nucleus sampling parameter (0-1, default: 1.0)
- `frequency_penalty` (FLOAT, required): Penalize frequent tokens (-2 to 2, default: 0)
- `presence_penalty` (FLOAT, required): Penalize present tokens (-2 to 2, default: 0)
- `system_prompt` (STRING, optional): System instructions for the model
- `images` (IMAGE, optional): Image input for vision models

**Outputs:**
- `response` (STRING): Generated text response

**Features:**
- Streaming output with color-coded console display
- Automatic image conversion to base64 for vision models
- Full parameter control for fine-tuned generation

---

#### GPT Image Generator Chat

Generate images using GPT vision models with prompt and reference image support.

**Inputs:**
- `prompt` (STRING, required): Description of the image to generate
- `model` (STRING, required): Model name (default: "gpt-4o-image-vip")
- `api_url` (STRING, required): API endpoint URL
- `api_key` (STRING, required): Your API key
- `ratio` (STRING, required): Image aspect ratio (1:1, 2:3, 3:2)
- `num_images` (STRING, required): Number of images to generate (1, 2, or 4)
- `seed` (INT, required): Random seed
- `images` (IMAGE, optional): Reference images for image-to-image generation

**Outputs:**
- `images` (IMAGE): List of generated images
- `API Respond` (STRING): Full API response text

**Features:**
- Automatic image downloading and tensor conversion
- Reference image support for guided generation
- Retry logic for robust image downloads
- Batch output support

---

### Ollama Nodes

#### Ollama Generate

Generate text with local Ollama models, supporting vision and thinking modes.

**Inputs:**
- `prompt` (STRING, required): The user prompt
- `url` (STRING, required): Ollama server URL (default: "http://127.0.0.1:11434")
- `model` (dropdown, required): Model name (dynamically loaded from Ollama)
- `seed` (INT, required): Random seed
- `temperature` (FLOAT, required): Sampling temperature (0-2, default: 0.8)
- `num_ctx` (INT, required): Context window size (1-8192, default: 2048)
- `num_predict` (INT, required): Max tokens to predict (-1 for unlimited, default: -1)
- `keep_alive` (INT, required): Keep model in memory (seconds, -1 for indefinite)
- `thinking` (STRING, required): Thinking mode (ON, OFF, HIGH, MEDIUM, LOW, NONE)
- `keep_reason` (BOOLEAN, required): Whether to keep thinking tags in output
- `format` (STRING, required): Output format (text, json, or empty)
- `images` (IMAGE, optional): Image batch for vision models

**Outputs:**
- `response` (STRING): Generated text response

**Features:**
- Streaming output with color-coded thinking (yellow) and response (green)
- Automatic GPU memory clearing before generation
- Thinking mode support for reasoning models
- Vision model support with batch image processing
- JSON mode for structured outputs

---

#### Ollama Chat

Multi-turn conversation with a single Ollama model.

**Inputs:**
- `system` (STRING, required): System prompt defining assistant behavior
- `prompt1` (STRING, required): First user message
- `prompt2` (STRING, optional): Second user message for round two
- `url` (STRING, required): Ollama server URL
- `model` (dropdown, required): Model name
- `seed` (INT, required): Random seed
- `temperature` (FLOAT, required): Sampling temperature (0-2, default: 0.8)
- `num_ctx` (INT, required): Context window size (default: 2048)
- `num_predict` (INT, required): Max tokens to predict (default: -1)
- `keep_alive` (INT, required): Keep model in memory (seconds, default: 1)
- `format` (STRING, required): Output format (text, json, or empty)

**Outputs:**
- `response1` (STRING): Response to first prompt
- `response2` (STRING): Response to second prompt (if provided)

**Features:**
- Two-round conversation in a single node
- Automatic thinking tag removal
- Context preservation between rounds
- Performance logging (tokens/second)

---

#### Ollama Chat Dual Round

Multi-turn conversation with two different models (useful for refinement workflows).

**Inputs:**
- `system` (STRING, required): System prompt
- `prompt1` (STRING, required): First user message
- `prompt2` (STRING, optional): Second user message
- `url` (STRING, required): Ollama server URL
- `model1` (dropdown, required): First model name
- `model2` (dropdown, required): Second model name
- `seed` (INT, required): Random seed
- `top_k` (INT, required): Top-k sampling (0-100, default: 40)
- `top_p` (FLOAT, required): Nucleus sampling (0-1, default: 0.9)
- `temperature` (FLOAT, required): Sampling temperature (default: 0.8)
- `num_ctx` (INT, required): Context window size (default: 2048)
- `num_predict` (INT, required): Max tokens to predict (default: -1)
- `tfs_z` (FLOAT, required): Tail free sampling (1-1000, default: 1)
- `keep_alive` (INT, required): Keep models in memory (seconds, default: 1)
- `keep_thinking` (BOOLEAN, required): Keep thinking tags in output
- `format` (STRING, required): Output format

**Outputs:**
- `response1` (STRING): First model's response
- `response2` (STRING): Second model's response (if prompt2 provided)

**Features:**
- Use different models for different conversation rounds
- Useful for draft → refinement workflows
- Advanced sampling parameter control
- Optional thinking tag preservation

---

### Text Processing Nodes

#### Text Template

Powerful Jinja2 template rendering with unlimited dynamic inputs.

**Inputs:**
- `template` (STRING, required): Jinja2 template text (default: "{{input_1}}")
- Dynamic inputs: Connect unlimited inputs which are auto-numbered (input_1, input_2, etc.)

**Outputs:**
- `rendered_text` (STRING): Rendered template

**Features:**
- Full Jinja2 syntax support (loops, conditionals, filters)
- Unlimited dynamic inputs via JavaScript extension
- Recursive template rendering (up to 10 iterations)
- Type-aware: handles strings, numbers, lists, dicts, booleans
- Built-in `now()` function for datetime formatting
- Helpful error messages with available variables
- Auto-numbering of connected inputs

**Example Templates:**
```jinja2
# Simple substitution
{{input_1}} is {{input_2}}

# Conditional
{% if input_1 %}Positive: {{input_1}}{% else %}No input{% endif %}

# Loops
{% for item in input_1 %}Item: {{item}}
{% endfor %}

# Complex formatting
Generate an image of {{input_1}} in {{input_2}} style,
with {{input_3}} lighting

# Datetime formatting (using Python's strftime format codes)
Current date: {{now().strftime('%Y-%m-%d')}}
Current time: {{now().strftime('%H:%M:%S')}}
Full datetime: {{now().strftime('%Y-%m-%d %H:%M:%S')}}
Custom format: {{now().strftime('%B %d, %Y at %I:%M %p')}}

# Using datetime in prompts
Photo taken on {{now().strftime('%B %d, %Y')}}, {{input_1}}
```

**Common datetime format codes:**
- `%Y` - Year with century (2024)
- `%m` - Month as number (01-12)
- `%d` - Day of month (01-31)
- `%H` - Hour 24-hour (00-23)
- `%I` - Hour 12-hour (01-12)
- `%M` - Minute (00-59)
- `%S` - Second (00-59)
- `%p` - AM/PM
- `%B` - Full month name (January)
- `%b` - Abbreviated month name (Jan)
- `%A` - Full weekday name (Monday)
- `%a` - Abbreviated weekday name (Mon)

---

#### Text Extract

Extract text between begin and end markers.

**Inputs:**
- `input_text` (STRING, required): Source text
- `begin` (STRING, optional): Start marker (omit to start from beginning)
- `end` (STRING, optional): End marker (omit to extract to end)

**Outputs:**
- `extracted_text` (STRING): Extracted portion

**Use Cases:**
- Extract JSON from markdown code blocks
- Parse specific sections from LLM responses
- Extract content between XML/HTML tags

---

#### Text Remove

Remove text between begin and end markers.

**Inputs:**
- `input_text` (STRING, required): Source text
- `begin` (STRING, required): Start marker (inclusive)
- `end` (STRING, required): End marker (inclusive)

**Outputs:**
- `text` (STRING): Text with section removed

**Use Cases:**
- Remove thinking tags from responses
- Strip code blocks or metadata
- Clean up formatted text

---

### SEGS Processing Nodes

#### SEGS Captioner

Automatically caption segmented regions using Ollama vision models.

**Inputs:**
- `segs` (SEGS, required): Segmentation data from detection nodes
- `prompt_template` (STRING, required): Caption prompt template
- `url` (STRING, required): Ollama server URL
- `model` (dropdown, required): Vision model name
- `seed` (INT, required): Random seed
- `temperature` (FLOAT, required): Sampling temperature (default: 0.8)
- `num_ctx` (INT, required): Context window size (default: 2048)
- `num_predict` (INT, required): Max tokens per caption (default: 500)
- `keep_alive` (INT, required): Keep model in memory (seconds)
- `thinking` (STRING, required): Thinking mode (ON, OFF, HIGH, MEDIUM, LOW, NONE)
- `keep_reason` (BOOLEAN, required): Whether to keep thinking tags in output
- `fallback_image_opt` (IMAGE, optional): Fallback image for cropping when SEGS lacks cropped images

**Outputs:**
- `wildcard_captions` (STRING): Formatted captions in [LAB] wildcard format

**Output Format:**
```
[LAB]
[1] description of first segment
[2] description of second segment
[3] description of third segment
```

**Features:**
- Automatic handling of SEGS cropped images or fallback image cropping
- Batch processing of all segments
- Memory optimization between captions
- Incremental seed for variety
- Clean single-line captions
- Compatible with LAB wildcard format

**Use Cases:**
- Auto-caption detected objects
- Generate descriptions for image regions
- Create training data labels
- Build dynamic prompts from segmented content

---

## Usage Examples

### Example 1: Simple Prompt Generation

```
[TextTemplate] → [OpenAI Generate]
Template: "Generate a creative prompt about {{input_1}}"
Input_1: "a futuristic city"
→ OpenAI → Response: "A sprawling metropolis..."
```

### Example 2: Multi-Turn Conversation

```
[TextTemplate] → [Ollama Chat]
Prompt1: "What is cyberpunk?"
Prompt2: "Give me 3 examples"
→ Response1: Definition
→ Response2: Examples list
```

### Example 3: Vision Analysis

```
[Load Image] → [OpenAI Generate]
Prompt: "Describe this image in detail"
→ Response: Detailed description
```

### Example 4: Draft and Refine

```
[TextTemplate] → [Ollama Chat Dual Round]
Model1: "llama3.2:1b" (fast draft)
Model2: "qwen2.5:14b" (quality refinement)
Prompt1: "Write a story about {{topic}}"
Prompt2: "Make it more dramatic"
```

### Example 5: Segment Captioning Pipeline

```
[Load Image] → [Detection Node] → [SEGS Captioner]
→ Wildcard Captions: [LAB] format ready for use
```

## Advanced Features

### Dynamic Model Loading

The Ollama nodes feature JavaScript-based dynamic model loading. Available models are fetched from your Ollama server and populated in the dropdown automatically.

### Memory Optimization

Before each Ollama generation, the nodes automatically:
1. Unload all ComfyUI models
2. Run Python garbage collection
3. Clear PyTorch CUDA cache
4. Wait for memory to settle

This ensures maximum available memory for local LLM inference.

### Thinking Mode

Ollama nodes support reasoning models with thinking capabilities:
- `NONE`: No thinking mode
- `ON/OFF`: Binary toggle
- `HIGH/MEDIUM/LOW`: Graduated thinking intensity

The `keep_reason` parameter controls whether thinking tags remain in the output.

### Streaming Output

All generation nodes use streaming APIs with color-coded console output:
- Yellow: Thinking/reasoning
- Green: Response text
- Performance stats logged after completion

## API Endpoints

The plugin exposes one API endpoint:

**POST /yallm/get_ollama_models**
- Request body: `{"url": "http://127.0.0.1:11434"}`
- Response: List of available model names
- Used by frontend for dynamic model dropdown population

## Configuration

All configuration is done through node parameters in ComfyUI's interface. No external configuration files required.

### Recommended Settings

**For Quality:**
- Temperature: 0.7-0.9
- Top P: 0.9-0.95
- Higher max_tokens

**For Consistency:**
- Temperature: 0.1-0.3
- Fixed seed value
- Top P: 0.9

**For Speed:**
- Smaller models
- Lower num_ctx
- Limited num_predict

## Troubleshooting

### Models not appearing in Ollama dropdowns
- Ensure Ollama server is running
- Check the URL is correct
- Verify models are installed: `ollama list`

### Out of memory errors
- Reduce num_ctx
- Use smaller models
- Ensure memory optimization is working
- Close other GPU applications

### Slow generation
- Use local Ollama for better performance
- Reduce num_ctx and max_tokens
- Consider smaller models
- Check network connectivity for API nodes

### Vision model not working
- Ensure model supports vision (e.g., llama3.2-vision, gpt-4o)
- Check image format and size
- Verify API supports vision endpoints

## Dependencies

- openai>=1.40.1
- ollama
- pillow
- numpy
- jinja2
- tenacity
- torch (provided by ComfyUI)
- aiohttp (provided by ComfyUI)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Credits

Developed for the ComfyUI community to provide flexible and powerful LLM integration.
 
