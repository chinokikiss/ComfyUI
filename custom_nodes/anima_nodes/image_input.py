import io
import base64
import numpy as np
from PIL import Image
import torch


class LoadImageFromBase64:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_data": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "Anima"

    def load_image(self, image_data):
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]
        return (img_tensor,)
