import jinja2
import re
from typing import Optional, Tuple


class TextTemplate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template": (
                    "STRING",
                    {"multiline": True, "default": "{{input1}}"},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rendered_text",)
    CATEGORY = "Yet Another LLM"
    FUNCTION = "render"

    def render(self, template: str, **kwargs):
        # Prepare all inputs from kwargs (dynamic inputs)
        inputs = {}
        for key, value in kwargs.items():
            if key.startswith("input"):
                # Handle both string and non-string types
                if isinstance(value, str):
                    inputs[key] = value.strip() if value else ""
                else:
                    inputs[key] = value if value is not None else ""

        # Recursive template rendering with limit
        max_iterations = 10
        current_text = template

        for iteration in range(max_iterations):
            # Check if there are any template tags remaining (now supports any input number)
            template_pattern = r'\{\{input\d+\}\}'
            if not re.search(template_pattern, current_text):
                # No more template tags found, we're done
                break

            # Render the current text
            template_env = jinja2.Environment(autoescape=False)
            template_str = template_env.from_string(current_text)

            # Set all inputs as globals
            for key, value in inputs.items():
                template_str.globals[key] = value

            current_text = template_str.render()

        # Check if we still have template tags after max iterations
        template_pattern = r'\{\{input\d+\}\}'
        if re.search(template_pattern, current_text):
            raise ValueError(
                f"Template rendering failed: still contains template tags after {max_iterations} iterations. "
                f"Possible infinite recursion detected or undefined inputs referenced."
            )

        return (current_text,)


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


NODE_CLASS_MAPPINGS = {
    "yaLLMTextTemplate": TextTemplate,
    "yaTextExtract": TextExtract,
    "yaTextRemove": TextRemove,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaLLMTextTemplate": "Text Template",
    "yaTextExtract": "Text Extract",
    "yaTextRemove": "Text Remove",
}
