from .nodes import LLMApiModelLoader, LLMChat, TextTemplate, TextExtract, TextRemove


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "yaLLMApiModelLoader": LLMApiModelLoader,
    "yaLLMChat": LLMChat,
    "yaLLMTextTemplate": TextTemplate,
    "yaTextExtract": TextExtract,
    "yaTextRemove": TextRemove,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "yaLLMApiModelLoader": "Load API Model",
    "yaLLMChat": "API LLM Chat",
    "yaLLMTextTemplate": "Text Template",
    "yaTextExtract": "Text Extract",
    "yaTextRemove": "Text Strip",
}
