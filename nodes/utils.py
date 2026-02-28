import torch
import ast
import gc
import comfy.model_management

# Common ComfyUI data types for the "any" input
COMFY_ANY_TYPE = ["IMAGE", "MASK", "LATENT", "MODEL", "CONDITIONING", "CROP_DATA", "*"]


class MidnightLook_AnyToString:
    """Converts any ComfyUI data type to a human-readable string."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_data": (COMFY_ANY_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "any_to_string"
    CATEGORY = "MidnightLook/Utils"

    def any_to_string(self, input_data):
        if isinstance(input_data, torch.Tensor):
            output_string = (
                f"Tensor Shape: {input_data.shape}, "
                f"DType: {input_data.dtype}, "
                f"Device: {input_data.device}"
            )
        else:
            output_string = str(input_data)

        print(f"✅ MidnightLook (AnyToString): type='{type(input_data).__name__}' → {output_string}")
        return (output_string,)


class MidnightLook_StringToBBOX:
    """
    Parses a string like ``"((322, 322), (330, 174, 652, 496))"``
    and returns the second tuple as a BBOX tensor ``[330, 174, 652, 496]``.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data_string": ("STRING", {
                    "multiline": False,
                    "default": "((0, 0), (0, 0, 512, 512))",
                }),
            },
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Utils"

    def process(self, data_string: str):
        try:
            parsed_data = ast.literal_eval(data_string)

            if not isinstance(parsed_data, tuple) or len(parsed_data) != 2:
                raise ValueError("Input string must be a tuple of two elements.")

            bbox_tuple = parsed_data[1]
            if not isinstance(bbox_tuple, tuple) or len(bbox_tuple) != 4:
                raise ValueError(
                    "The second element must be a tuple of four numbers (left, top, right, bottom)."
                )

            left, top, right, bottom = bbox_tuple
            bbox_tensor = torch.tensor([left, top, right, bottom], dtype=torch.int64)
            return ([bbox_tensor],)

        except (ValueError, SyntaxError) as e:
            print(f"ERROR: MidnightLook_StringToBBOX - Invalid input: {e}")
            print(f"    Input was: '{data_string}'")
            default_bbox = torch.tensor([0, 0, 512, 512], dtype=torch.int64)
            return ([default_bbox],)


class MidnightLook_CropDATAToBBOX:
    """Converts CROP_DATA (a tuple of crop coordinates) into a BBOX tensor."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_data": ("CROP_DATA",),
            },
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Utils"

    def process(self, crop_data):
        crop_info = crop_data[0]

        if not isinstance(crop_info, tuple) or len(crop_info) < 4:
            raise ValueError(
                "Invalid crop_data format. Expected a tuple with at least 4 elements."
            )

        left, top, right, bottom = crop_info[0], crop_info[1], crop_info[2], crop_info[3]
        bbox_tensor = torch.tensor([left, top, right, bottom], dtype=torch.int64)
        return ([bbox_tensor],)



class MidnightLook_DisplayAny:
    """Takes any input and displays it as a string in the UI."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": (COMFY_ANY_TYPE,),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "display_any"
    CATEGORY = "MidnightLook/Utils"
    OUTPUT_NODE = True

    def display_any(self, source):
        if isinstance(source, torch.Tensor):
            output_string = (
                f"Tensor Shape: {source.shape}, "
                f"DType: {source.dtype}, "
                f"Device: {source.device}"
            )
        else:
            try:
                output_string = str(source)
            except Exception as e:
                output_string = f"Error converting to string: {e}"

        return {"ui": {"text": [output_string]}}


class MidnightLook_VRAMClear:
    """
    Clears VRAM by unloading models and running garbage collection.
    Passes the input through unchanged.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_data": ("*",), # Wildcard type
                "mode": (["Specific Object", "All Models", "Nothing"], {"default": "Specific Object"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output_data",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Utils"

    def process(self, input_data, mode):
        if mode == "Nothing":
            return (input_data,)

        print(f"🧹 MidnightLook VRAM Clear: Mode = {mode}")
        
        # 1. Handle All Models unloading
        if mode == "All Models":
            print("   Unloading ALL models...")
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            
        # 2. Handle specific object unloading (if possible)
        elif mode == "Specific Object":
            # If input is a model wrapper (ComfyUI often wraps models)
            # Try to unload patches or models associated with input
            pass

        # 3. Always run GC and Empty Cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        return (input_data,)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "MidnightLook_AnyToString": MidnightLook_AnyToString,
    "MidnightLook_StringToBBOX": MidnightLook_StringToBBOX,
    "MidnightLook_CropDATAToBBOX": MidnightLook_CropDATAToBBOX,
    "MidnightLook_DisplayAny": MidnightLook_DisplayAny,
    "MidnightLook_VRAMClear": MidnightLook_VRAMClear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_AnyToString": "Any to String (ML)",
    "MidnightLook_StringToBBOX": "String to BBox (ML)",
    "MidnightLook_CropDATAToBBOX": "Crop Data to BBox (ML)",
    "MidnightLook_DisplayAny": "Display Any (ML)",
    "MidnightLook_VRAMClear": "Clear VRAM (Pass-through)",
}
