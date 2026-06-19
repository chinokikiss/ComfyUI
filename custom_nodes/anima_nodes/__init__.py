from .image_output import ImageDataOutput
from .image_input import LoadImageFromBase64

NODE_CLASS_MAPPINGS = {
    "ImageDataOutput": ImageDataOutput,
    "LoadImageFromBase64": LoadImageFromBase64,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
