class MidnightLook_TextBox:
    """
    A simple text box node for writing and managing multi-line text.
    Has no inputs, and outputs a string.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Utils"
    
    # OUTPUT_NODE allows this node to show up or handle UI natively if needed,
    # though usually custom string nodes operate without issue.

    def process(self, text):
        return (text,)
