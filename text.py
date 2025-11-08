import jinja2
import re
from typing import Optional, Tuple


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
        # Prepare all inputs
        inputs = {
            "input1": input1.strip() if input1 is not None and input1 != "" else "",
            "input2": input2.strip() if input2 is not None and input2 != "" else "",
            "input3": input3.strip() if input3 is not None and input3 != "" else "",
            "input4": input4.strip() if input4 is not None and input4 != "" else "",
            "input5": input5.strip() if input5 is not None and input5 != "" else "",
            "input6": input6.strip() if input6 is not None and input6 != "" else "",
            "input7": input7.strip() if input7 is not None and input7 != "" else "",
            "input8": input8.strip() if input8 is not None and input8 != "" else "",
        }

        # Recursive template rendering with limit
        max_iterations = 10
        current_text = template

        for iteration in range(max_iterations):
            # Check if there are any template tags remaining
            template_pattern = r'\{\{input[1-8]\}\}'
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
        template_pattern = r'\{\{input[1-8]\}\}'
        if re.search(template_pattern, current_text):
            raise ValueError(f"Template rendering failed: still contains template tags after {max_iterations} iterations. Possible infinite recursion detected.")

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