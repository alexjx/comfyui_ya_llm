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
                    {"multiline": True, "default": "{{input_1}}"},
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
            # Preserve string stripping for strings while passing other types as-is
            # Jinja2 natively handles numbers, lists, dicts, booleans, etc.
            if value is None:
                inputs[key] = ""
            elif isinstance(value, str):
                inputs[key] = value.strip() if value else ""
            else:
                # Let Jinja2 handle non-string types directly
                inputs[key] = value

        # Recursive template rendering with limit
        max_iterations = 10
        current_text = template

        for iteration in range(max_iterations):
            # Check if there are any template tags remaining
            # Support both {{inputN}} and custom renamed variables
            template_pattern = r'\{\{[^}]+\}\}'
            matches = re.findall(template_pattern, current_text)
            if not matches:
                # No more template tags found, we're done
                break

            # Render the current text
            template_env = jinja2.Environment(autoescape=False)
            template_str = template_env.from_string(current_text)

            # Set all inputs as globals (includes both original and renamed names)
            for key, value in inputs.items():
                template_str.globals[key] = value

            try:
                current_text = template_str.render()
            except jinja2.exceptions.UndefinedError as e:
                # Provide helpful error message if a variable is used but not defined
                raise ValueError(
                    f"Template rendering failed: {str(e)}. "
                    f"Available variables: {', '.join(sorted(inputs.keys()))}"
                )

        # Check if we still have unresolved template tags after max iterations
        remaining_matches = re.findall(r'\{\{[^}]+\}\}', current_text)
        if remaining_matches:
            raise ValueError(
                f"Template rendering failed: still contains template tags {remaining_matches} after {max_iterations} iterations. "
                f"Possible infinite recursion detected or undefined inputs referenced. "
                f"Available variables: {', '.join(sorted(inputs.keys()))}"
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
                "marker_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_text",)
    CATEGORY = "Yet Another LLM"
    FUNCTION = "extract"

    def extract(
        self, input_text: str, begin: Optional[str] = None, end: Optional[str] = None,
        marker_mode: bool = False
    ) -> Tuple[str]:
        if begin:
            start_idx = input_text.find(begin)
            if start_idx == -1:
                raise ValueError(f"Begin string '{begin}' not found in input text")
            # If marker_mode is enabled, skip past the begin marker
            if marker_mode:
                start_idx += len(begin)
        else:
            start_idx = 0
        if end:
            end_idx = input_text.find(end, start_idx)
            if end_idx == -1:
                raise ValueError(f"End string '{end}' not found in input text")
            # If marker_mode is disabled, include the end marker
            if not marker_mode:
                end_idx += len(end)
        else:
            end_idx = len(input_text)

        # Extract and strip whitespace
        result = input_text[start_idx:end_idx]
        if marker_mode:
            result = result.strip()

        return (result,)


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
