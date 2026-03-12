# nodes/__init__.py — Central registry for all MidnightLook nodes
#
# To add a new node:
#   1. Create or edit a submodule in this package
#   2. Define NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS in it
#   3. Import the submodule below

from .utils import (
    NODE_CLASS_MAPPINGS as _utils_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _utils_name,
)
from .inpaint import (
    NODE_CLASS_MAPPINGS as _inpaint_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _inpaint_name,
)
from .image import (
    NODE_CLASS_MAPPINGS as _image_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _image_name,
)
from .latent import (
    NODE_CLASS_MAPPINGS as _latent_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _latent_name,
)
from .image_compare import MidnightLook_ImageCompare


from .mediapipe_crop_node import (
    NODE_CLASS_MAPPINGS as _mp_crop_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _mp_crop_name,
)
from .deepface_node import (
    NODE_CLASS_MAPPINGS as _df_crop_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _df_crop_name,
)
from .qwen2_5_vl import (
    NODE_CLASS_MAPPINGS as _qwen_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _qwen_name,
)
from .z_image_prompt import (
    NODE_CLASS_MAPPINGS as _z_prompt_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _z_prompt_name,
)
from .preset_prompt import MidnightLook_PresetPrompt
from .text_box import MidnightLook_TextBox
from .loop_control_nodes import (
    NODE_CLASS_MAPPINGS as _loop_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _loop_name,
)
from .midnight_detailer import (
    NODE_CLASS_MAPPINGS as _detailer_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _detailer_name,
)
from .iterative_upscale import (
    NODE_CLASS_MAPPINGS as _upscale_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _upscale_name,
)
from .url_lora_loader import (
    NODE_CLASS_MAPPINGS as _url_lora_cls,
    NODE_DISPLAY_NAME_MAPPINGS as _url_lora_name,
)

NODE_CLASS_MAPPINGS = {
    **_utils_cls,
    **_inpaint_cls,
    **_image_cls,
    **_latent_cls,
    "MidnightLook_ImageCompare": MidnightLook_ImageCompare,


    **_mp_crop_cls,
    **_df_crop_cls,
    **_qwen_cls,
    **_z_prompt_cls,
    "MidnightLook_PresetPrompt": MidnightLook_PresetPrompt,
    "MidnightLook_TextBox": MidnightLook_TextBox,
    **_loop_cls,
    **_detailer_cls,
    **_upscale_cls,
    **_url_lora_cls,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_utils_name,
    **_inpaint_name,
    **_image_name,
    **_latent_name,
    "MidnightLook_ImageCompare": "Image Compare (ML)",


    **_mp_crop_name,
    **_df_crop_name,
    **_qwen_name,
    **_z_prompt_name,
    "MidnightLook_PresetPrompt": "Preset Prompt (ML)",
    "MidnightLook_TextBox": "Text Box (ML)",
    **_loop_name,
    **_detailer_name,
    **_upscale_name,
    **_url_lora_name,
}
