import jinja2
import re
from typing import Optional, Tuple
from datetime import datetime


def _extract_char_index(error_message: str) -> Optional[int]:
    match = re.search(r"\bat\s+(\d+)\b", error_message)
    if not match:
        return None
    return int(match.group(1))


def _line_and_column_from_index(text: str, index: int) -> Tuple[int, int]:
    safe_index = min(max(index, 0), len(text))
    line_no = text.count("\n", 0, safe_index) + 1
    line_start = text.rfind("\n", 0, safe_index) + 1
    column_no = safe_index - line_start + 1
    return (line_no, column_no)


def _line_content(text: str, line_no: int) -> str:
    lines = text.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


def _try_decode_json_escapes(text: str) -> str:
    """
    Reverse JSON encoding that may have been applied to template strings.

    When workflow data is serialized through multiple layers (e.g., ComfyUI's
    internal serialization + JSON), the template string can become double-encoded.
    This manifests as '\\u0027' for single quotes and '\\\"' for double quotes
    appearing in the string, which breaks Jinja2 template parsing.

    This function reverses the JSON encoding by replacing known JSON escape
    sequences with their actual characters.

    Returns the decoded string if any JSON escapes were found and reversed,
    otherwise returns the original string unchanged.
    """
    # Check if the string contains JSON escape patterns
    has_u0027 = "\\u0027" in text
    has_escaped_quote = '\\"' in text

    if not has_u0027 and not has_escaped_quote:
        return text

    # Replace JSON escape sequences with their actual characters
    # \u0027 is the JSON escape for single quote (')
    # \" is the JSON escape for double quote (")
    # We also handle other common JSON escapes that might appear
    result = text
    if has_u0027:
        result = result.replace("\\u0027", "'")
    if has_escaped_quote:
        result = result.replace('\\"', '"')

    return result


def _format_template_syntax_error(
    error: jinja2.exceptions.TemplateSyntaxError,
    source_text: str,
    inputs: dict,
    iteration: int,
) -> str:
    message = str(error)
    char_index = getattr(error, "pos", None)
    if char_index is None:
        char_index = _extract_char_index(message)

    line_no = getattr(error, "lineno", None)
    column_no = None
    if char_index is not None:
        line_no, column_no = _line_and_column_from_index(source_text, char_index)

    context_line = _line_content(source_text, line_no) if line_no else ""
    pointer_line = ""
    if context_line and column_no is not None and column_no > 0:
        pointer_line = " " * (column_no - 1) + "^"

    available_inputs = ", ".join(sorted(inputs.keys())) if inputs else "(none)"

    details = [f"Template syntax error (iteration {iteration + 1}): {message}"]
    if line_no:
        details.append(f"Line {line_no}")
    if column_no:
        details.append(f"Column {column_no}")
    if char_index is not None:
        details.append(f"Character index {char_index}")
    if context_line:
        details.append("Context:")
        details.append(context_line)
        if pointer_line:
            details.append(pointer_line)
    details.append(f"Available variables: {available_inputs}")
    return "\n".join(details)


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
        current_text = _try_decode_json_escapes(template)

        for iteration in range(max_iterations):
            # Decode any JSON escapes that may have been introduced (e.g., via ComfyUI serialization)
            current_text = _try_decode_json_escapes(current_text)

            # Check if there are any template tags remaining
            # Match {{...}}, {%...%}, and {#...#} (variables, control structures, comments)
            template_pattern = r"\{[{%#][^}]+[}%#]\}"
            matches = re.findall(template_pattern, current_text)
            if not matches:
                # No more template tags found, we're done
                break

            # Render the current text
            template_env = jinja2.Environment(autoescape=False)
            template_env.filters["boolean"] = bool
            try:
                template_str = template_env.from_string(current_text)
            except jinja2.exceptions.TemplateSyntaxError as e:
                raise ValueError(
                    _format_template_syntax_error(e, current_text, inputs, iteration)
                ) from e

            # Set all inputs as globals (includes both original and renamed names)
            for key, value in inputs.items():
                template_str.globals[key] = value

            # Add utility functions
            template_str.globals["now"] = lambda: datetime.now()
            template_str.globals["bool"] = bool

            try:
                current_text = template_str.render()
            except jinja2.exceptions.UndefinedError as e:
                # Provide helpful error message if a variable is used but not defined
                raise ValueError(
                    f"Template rendering failed: {str(e)}. "
                    f"Available variables: {', '.join(sorted(inputs.keys()))}"
                ) from e
            except jinja2.exceptions.TemplateSyntaxError as e:
                raise ValueError(
                    _format_template_syntax_error(e, current_text, inputs, iteration)
                ) from e

        # Check if we still have unresolved template tags after max iterations
        remaining_matches = re.findall(r"\{[{%#][^}]+[}%#]\}", current_text)
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
        self,
        input_text: str,
        begin: Optional[str] = None,
        end: Optional[str] = None,
        marker_mode: bool = False,
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


class ImageLister:
    """Utility node to collect multiple images into a list"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},  # Dynamic inputs added via JavaScript
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "Yet Another LLM"
    FUNCTION = "list_images"

    def list_images(self, **kwargs):
        """Collect all image inputs and return as a list"""
        image_list = []

        # Collect images from image_1, image_2, image_3, image_4
        for key in sorted(kwargs.keys()):
            if key.startswith("image_"):
                img = kwargs[key]
                if img is not None:
                    image_list.append(img)

        return (image_list,)


NODE_CLASS_MAPPINGS = {
    "yaLLMTextTemplate": TextTemplate,
    "yaTextExtract": TextExtract,
    "yaTextRemove": TextRemove,
    "yaImageLister": ImageLister,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yaLLMTextTemplate": "Text Template",
    "yaTextExtract": "Text Extract",
    "yaTextRemove": "Text Remove",
    "yaImageLister": "Image Lister",
}
