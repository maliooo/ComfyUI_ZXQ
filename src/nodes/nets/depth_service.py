import requests
import json
import base64
from PIL import Image
import io
import torch
import numpy as np
import time
from rich import print
from .base_service import BaseService

model_name_2_model_name = {
    "depth_anything_v2_large": "large",
    "depth_anything_v2_indoor": "indoor",
    "depth_anything_v2_outdoor": "outdoor"
}

class DepthAnythingService(BaseService):
    """
    通过调用Depth-Anything-V2深度估计服务实现图像深度估计的ComfyUI节点。

    输入:
        image_base64: Base64编码的输入图像字符串。
        service_url: Depth-Anything-V2深度估计服务的URL。
        model_name: 使用的模型名称 (large/indoor/outdoor)。
        short_size: 图像预处理时调整大小的短边长度。

    输出:
        depth_image: 深度估计图像 (IMAGE a.k.a. Tensor)。
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base64": ("STRING", ),
                "service_url": ("STRING", {"default": "http://172.31.48.6:28003/estimate_depth_base64"}),
                "model_name": (["depth_anything_v2_large", "depth_anything_v2_indoor", "depth_anything_v2_outdoor"], {"default": "depth_anything_v2_large"}),
                "short_size": ("INT", {"default": 512, "min": 1, "max": 1536, "tooltip": "图片短边长度，用于调整图片大小"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_image",)
    FUNCTION = "estimate_depth"
    CATEGORY = "ZXQ/Depth"

    _NODE_NAME = "深度估计服务-depth"
    _NODE_ZXQ = "深度估计服务-depth"
    DESCRIPTION = "调用Depth-Anything-V2深度估计服务"

    def estimate_depth(self, image_base64, service_url, model_name, short_size):
        start_time = time.time()
        service_url = self.ip_map(service_url)  # 替换url
        if not image_base64:
            raise ValueError("image_base64 is required.")

        payload = {
            "image_url_or_base64": image_base64,
            "model_name": model_name_2_model_name[model_name],
            "short_size": short_size
        }

        headers = {
            "Content-Type": "application/json"
        }


        try:
            response = requests.post(service_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            # print(f"请求深度估计服务，返回结果: {result}")

            depth_image_b64 = result.get('depth_image')
            if not depth_image_b64:
                raise ValueError("Response does not contain 'depth_image'")
            
            image_data = base64.b64decode(depth_image_b64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            depth_image_tensor = torch.from_numpy(image_np)[None,]

            end_time = time.time()
            print(f"zxq深度depth服务，处理时间: {(end_time - start_time):.2f} seconds")

            return (depth_image_tensor,)

        except requests.exceptions.RequestException as e:
            error_message = f"zxq深度depth服务错误: API request failed: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        except Exception as e:
            error_message = f"zxq深度depth服务错误: An unexpected error occurred: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)


NODE_CLASS_MAPPINGS = {
    "DepthAnythingService": DepthAnythingService,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingService": "深度估计服务",
}
