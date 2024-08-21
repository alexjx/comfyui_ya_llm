from .nodes import LLMApiModelLoader, LLMChat


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "yaLLMApiModelLoader": LLMApiModelLoader,
    "yaLLMChat": LLMChat,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "yaLLMApiModelLoader": "Load API Model",
    "yaLLMChat": "API LLM Chat",
}
