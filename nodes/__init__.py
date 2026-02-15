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

NODE_CLASS_MAPPINGS = {
    **_utils_cls,
    **_inpaint_cls,
    **_image_cls,
    **_latent_cls,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_utils_name,
    **_inpaint_name,
    **_image_name,
    **_latent_name,
}
