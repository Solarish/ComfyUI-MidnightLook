# ComfyUI-MidnightLook — Custom nodes by MidnightLook
#
# All nodes live in the `nodes` sub-package.
# To add a new node, create or edit a submodule there and register it
# in nodes/__init__.py.

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("✅ MidnightLook: Custom nodes loaded successfully.")
