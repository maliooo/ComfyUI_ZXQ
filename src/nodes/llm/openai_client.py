import openai
import base64
import json
from typing import Optional, List, Dict, Any

class OpenAITextModel:
    """
    OpenAI文本模型调用

    输入：
        api_key: OpenAI API密钥
        model: 模型名称 (如 gpt-3.5-turbo, gpt-4)
        prompt: 输入提示词
        max_tokens: 最大生成token数
        temperature: 生成温度 (0-2)
        system_prompt: 系统提示词
    输出：
        response_text: 生成的文本响应
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"tooltip": "OpenAI API密钥"}),
                "model": ("STRING", {"default": "gpt-3.5-turbo", "tooltip": "模型名称"}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "输入提示词"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4000, "tooltip": "最大生成token数"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "tooltip": "生成温度"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "你是一个有用的AI助手", "tooltip": "系统提示词"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_ZXQ = "OpenAI Text Model 文本模型调用"
    DESCRIPTION = "调用OpenAI文本模型生成文本响应"
    CATEGORY = "ZXQ/OpenAI/文本模型"

    def forward(self, api_key: str, model: str, prompt: str, max_tokens: int, temperature: float, system_prompt: str = "你是一个有用的AI助手", **kwargs):
        try:
            # 设置API密钥
            client = openai.OpenAI(api_key=api_key)
            
            # 构建消息列表
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 调用API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 提取响应文本
            response_text = response.choices[0].message.content
            return (response_text,)
            
        except Exception as e:
            error_msg = f"OpenAI API调用失败: {str(e)}"
            return (error_msg,)


class OpenAIVisionModel:
    """
    OpenAI图片理解模型

    输入：
        api_key: OpenAI API密钥
        model: 模型名称 (如 gpt-4-vision-preview)
        prompt: 图片分析提示词
        image: 输入图片
        max_tokens: 最大生成token数
        temperature: 生成温度 (0-2)
    输出：
        response_text: 图片分析结果文本
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"tooltip": "OpenAI API密钥"}),
                "model": ("STRING", {"default": "gpt-4-vision-preview", "tooltip": "模型名称"}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "图片分析提示词"}),
                "image": ("IMAGE", {"tooltip": "输入图片"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4000, "tooltip": "最大生成token数"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "tooltip": "生成温度"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_ZXQ = "OpenAI Vision Model 图片理解模型"
    DESCRIPTION = "调用OpenAI图片理解模型分析图片内容"
    CATEGORY = "ZXQ/OpenAI/图片理解"

    def encode_image_to_base64(self, image):
        """将图片编码为base64字符串"""
        try:
            # 这里需要根据实际的图片格式进行编码
            # 假设image是numpy数组格式
            import cv2
            import base64
            import io
            
            # 转换为RGB格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # 编码为JPEG格式的base64
            _, buffer = cv2.imencode('.jpg', image_rgb)
            base64_string = base64.b64encode(buffer).decode('utf-8')
            return base64_string
        except Exception as e:
            raise Exception(f"图片编码失败: {str(e)}")

    def forward(self, api_key: str, model: str, prompt: str, image, max_tokens: int, temperature: float, **kwargs):
        try:
            # 设置API密钥
            client = openai.OpenAI(api_key=api_key)
            
            # 编码图片
            base64_image = self.encode_image_to_base64(image)
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # 调用API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 提取响应文本
            response_text = response.choices[0].message.content
            return (response_text,)
            
        except Exception as e:
            error_msg = f"OpenAI Vision API调用失败: {str(e)}"
            return (error_msg,)


class OpenAIImageEditModel:
    """
    OpenAI图片编辑生成模型

    输入：
        api_key: OpenAI API密钥
        model: 模型名称 (如 dall-e-3)
        prompt: 图片生成提示词
        size: 图片尺寸 (如 1024x1024, 1792x1024, 1024x1792)
        quality: 图片质量 (standard, hd)
        style: 图片风格 (vivid, natural)
        n: 生成图片数量
    输出：
        image_urls: 生成的图片URL列表 (JSON字符串)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"tooltip": "OpenAI API密钥"}),
                "model": ("STRING", {"default": "dall-e-3", "tooltip": "模型名称"}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "图片生成提示词"}),
                "size": (["1024x1024", "1792x1024", "1024x1792"], {"tooltip": "图片尺寸"}),
                "quality": (["standard", "hd"], {"tooltip": "图片质量"}),
                "style": (["vivid", "natural"], {"tooltip": "图片风格"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "生成图片数量"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_urls",)
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_ZXQ = "OpenAI Image Edit Model 图片编辑生成模型"
    DESCRIPTION = "调用OpenAI图片生成模型创建新图片"
    CATEGORY = "ZXQ/OpenAI/图片生成"

    def forward(self, api_key: str, model: str, prompt: str, size: str, quality: str, style: str, n: int, **kwargs):
        try:
            # 设置API密钥
            client = openai.OpenAI(api_key=api_key)
            
            # 调用图片生成API
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=n
            )
            
            # 提取图片URL
            image_urls = [image.url for image in response.data]
            
            # 返回JSON格式的URL列表
            result = {
                "image_urls": image_urls,
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "style": style,
                "count": len(image_urls)
            }
            
            return (json.dumps(result, indent=4, ensure_ascii=False),)
            
        except Exception as e:
            error_msg = f"OpenAI Image Generation API调用失败: {str(e)}"
            return (error_msg,)


if __name__ == "__main__":
    print("OpenAI Text Model INPUT_TYPES:", OpenAITextModel.INPUT_TYPES())
    print("OpenAI Vision Model INPUT_TYPES:", OpenAIVisionModel.INPUT_TYPES())
    print("OpenAI Image Edit Model INPUT_TYPES:", OpenAIImageEditModel.INPUT_TYPES())
