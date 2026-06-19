import io
import base64
import numpy as np
from PIL import Image
import torch


class ImageDataOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "Anima"

    def send_images(self, images):
        results = []
        for img_tensor in images:
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            results.append({"data": b64, "format": "png"})
        return {"ui": {"images_data": results}}
