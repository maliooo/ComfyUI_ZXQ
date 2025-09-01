import os
import base64
from pathlib import Path
from typing import Union, List, Dict, Optional
from openai import OpenAI
from PIL import Image
import io
import shutil
import traceback
from tqdm import tqdm
import torch
import numpy as np
from .config import SUPPORTED_MODELS, OPENROUTE_CONFIG
import sys

# 你的其他代码
print("程序开始运行...")

class OpenRouteImageAnalysis:
    """
    OpenRoute图片分析节点
    支持URL图片和本地图片的分析，调用OpenRoute API进行图片理解
    """
    
    @classmethod
    def INPUT_TYPES(s):
        default_model = OPENROUTE_CONFIG["default_model"]
        default_max_tokens = OPENROUTE_CONFIG["default_max_tokens"]
        default_temperature = OPENROUTE_CONFIG["default_temperature"]
        default_max_width = OPENROUTE_CONFIG["default_max_width"]
        default_max_height = OPENROUTE_CONFIG["default_max_height"]
        default_quality = OPENROUTE_CONFIG["default_quality"]
        default_base_url = OPENROUTE_CONFIG["base_url"]
        default_site_url = OPENROUTE_CONFIG["site_url"]
        default_site_name = OPENROUTE_CONFIG["site_name"]
        default_api_key = OPENROUTE_CONFIG["api_key"]

        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "请分析这张图片的内容",
                    "multiline": True,
                    "tooltip": "分析图片的提示词"
                }),
                "api_key": ("STRING", {
                    "default": default_api_key,
                    "tooltip": "OpenRoute API密钥"
                }),
                "model": (SUPPORTED_MODELS, {
                    "default": default_model,
                    "tooltip": "使用的模型名称"
                }),
                "max_tokens": ("INT", {
                    "default": default_max_tokens,
                    "min": 100,
                    "max": 4000,
                    "step": 100,
                    "tooltip": "最大输出token数"
                }),
                "temperature": ("FLOAT", {
                    "default": default_temperature,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "生成温度，控制随机性"
                }),
                "max_width": ("INT", {
                    "default": default_max_width,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "图片最大宽度（像素）"
                }),
                "max_height": ("INT", {
                    "default": default_max_height,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "图片最大高度（像素）"
                }),
                "quality": ("INT", {
                    "default": default_quality,
                    "min": 50,
                    "max": 100,
                    "step": 5,
                    "tooltip": "JPEG质量（1-100）"
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": default_base_url,
                    "tooltip": "API基础URL"
                }),
                "site_url": ("STRING", {
                    "default": default_site_url,
                    "tooltip": "网站URL（用于OpenRouter排名）"
                }),
                "site_name": ("STRING", {
                    "default": default_site_name,
                    "tooltip": "网站名称（用于OpenRouter排名）"
                }),
                "output_filename": ("STRING", {
                    "default": "generated_image.png",
                    "tooltip": "输出图片文件名"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("analysis_result", "generated_image",)
    FUNCTION = "execute"

    _NODE_ZXQ = "OpenRoute图片分析 OpenRoute Image Analysis"
    DESCRIPTION = "使用OpenRoute API分析图片内容，支持多种AI模型进行图片理解和生成"
    CATEGORY = "ZXQ/LLM/图片分析"

    def execute(self, image, prompt, api_key, model, max_tokens, temperature, max_width, max_height, quality, 
                base_url=None, site_url=None, site_name=None, 
                output_filename="generated_image.png"):
        """
        执行图片分析
        
        Args:
            image: 输入的图片张量
            prompt: 分析提示词
            api_key: OpenRoute API密钥
            model: 使用的模型名称
            max_tokens: 最大输出token数
            temperature: 生成温度
            max_width: 图片最大宽度
            max_height: 图片最大高度
            quality: JPEG质量
            base_url: API基础URL
            site_url: 网站URL
            site_name: 网站名称
            output_filename: 输出图片文件名
            
        Returns:
            analysis_result: 分析结果文本
            generated_image: 生成的图片（如果有的话）
        """
        # 使用配置文件中的默认值
        if base_url is None:
            base_url = OPENROUTE_CONFIG["base_url"]
        if site_url is None:
            site_url = OPENROUTE_CONFIG["site_url"]
        if site_name is None:
            site_name = OPENROUTE_CONFIG["site_name"]
            
        try:
            # 初始化OpenAI客户端
            client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            
            # 将torch张量转换为PIL图片
            pil_image = self._tensor_to_pil(image)
            
            # 调整图片尺寸
            pil_image = self._resize_image(pil_image, max_width, max_height)
            
            # 编码为base64
            base64_image = self._encode_image_to_base64(pil_image, quality)
            
            # 创建图片内容
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            
            # 创建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        image_content
                    ]
                }
            ]
            
            # 调用API
            print(f"🤖 正在使用 {model} 模型分析图片...")
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 获取分析结果
            analysis_result = completion.choices[0].message.content
            print(f"✅ 图片分析完成")
            
            # 检查是否有生成的图片
            generated_image = None
            if hasattr(completion.choices[0].message, 'images') and completion.choices[0].message.images:
                print(f"🎨 检测到生成的图片，正在处理...")
                generated_image = self._process_generated_images(completion.choices[0].message.images, output_filename)
            
            return (analysis_result, generated_image if generated_image is not None else image)
            
        except Exception as e:
            error_msg = f"图片分析失败: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return (error_msg, image)
    
    def _tensor_to_pil(self, image_tensor):
        """将torch张量转换为PIL图片"""
        # 确保图片张量格式正确 (batch, height, width, channels)
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]  # 取第一个batch
        
        # 转换为numpy数组并确保值在0-255范围内
        if image_tensor.dtype == torch.float32:
            image_array = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_array = image_tensor.cpu().numpy().astype(np.uint8)
        
        # 创建PIL图片
        pil_image = Image.fromarray(image_array)
        return pil_image
    
    def _resize_image(self, pil_image, max_width, max_height):
        """调整图片尺寸，保持比例"""
        original_width, original_height = pil_image.size
        
        # 计算缩放比例
        if max_width and max_height:
            scale_factor = min(max_width / original_width, max_height / original_height)
        elif max_width:
            scale_factor = max_width / original_width
        elif max_height:
            scale_factor = max_height / original_height
        else:
            return pil_image
        
        # 如果图片已经小于限制尺寸，则不缩放
        if scale_factor >= 1.0:
            print(f"📏 图片尺寸 {original_width}x{original_height} 无需缩放")
            return pil_image
        
        # 计算新尺寸
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # 缩放图片
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"📏 图片已缩放: {original_width}x{original_height} -> {new_width}x{new_height}")
        
        return resized_image
    
    def _encode_image_to_base64(self, pil_image, quality):
        """将PIL图片编码为base64字符串"""
        # 确保图片是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为字节流
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
        img_byte_arr.seek(0)
        
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    def _process_generated_images(self, images_data, output_filename):
        """处理API返回的生成图片"""
        try:
            for i, image_data in enumerate(images_data):
                if 'image_url' in image_data and 'url' in image_data['image_url']:
                    base64_string = image_data['image_url']['url']
                    
                    # 提取base64数据
                    if ',' in base64_string:
                        base64_data = base64_string.split(',')[1]
                    else:
                        base64_data = base64_string
                    
                    # 解码为图片字节
                    image_bytes = base64.b64decode(base64_data)
                    
                    # 创建PIL图片
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # 转换为RGB模式
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # 转换为torch张量
                    image_array = np.array(pil_image)
                    image_tensor = torch.from_numpy(image_array).float() / 255.0
                    
                    # 添加batch维度
                    image_tensor = image_tensor.unsqueeze(0)
                    
                    print(f"🎨 成功处理生成的图片 {i+1}")
                    return image_tensor
            
            print("⚠️ 未找到有效的生成图片数据")
            return None
            
        except Exception as e:
            print(f"❌ 处理生成图片失败: {e}")
            return None

# 批量处理节点
class OpenRouteBatchImageAnalysis:
    """
    OpenRoute批量图片分析节点
    支持批量处理多张图片
    """
    
    @classmethod
    def INPUT_TYPES(s):
        default_model = OPENROUTE_CONFIG["default_model"]
        default_max_tokens = OPENROUTE_CONFIG["default_max_tokens"]
        default_temperature = OPENROUTE_CONFIG["default_temperature"]
        default_base_url = OPENROUTE_CONFIG["base_url"]
        default_api_key = OPENROUTE_CONFIG["api_key"]
        
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "请分析这张图片的内容",
                    "multiline": True,
                    "tooltip": "分析图片的提示词"
                }),
                "api_key": ("STRING", {
                    "default": default_api_key,
                    "tooltip": "OpenRoute API密钥"
                }),
                "model": ("STRING", {
                    "default": default_model,
                    "tooltip": "使用的模型名称"
                }),
                "max_tokens": ("INT", {
                    "default": default_max_tokens,
                    "min": 100,
                    "max": 4000,
                    "step": 100,
                    "tooltip": "最大输出token数"
                }),
                "temperature": ("FLOAT", {
                    "default": default_temperature,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "生成温度，控制随机性"
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": default_base_url,
                    "tooltip": "API基础URL"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("batch_results",)
    FUNCTION = "execute"

    _NODE_ZXQ = "OpenRoute批量图片分析 OpenRoute Batch Image Analysis"
    DESCRIPTION = "批量使用OpenRoute API分析多张图片内容"
    CATEGORY = "ZXQ/LLM/图片分析"

    def execute(self, images, prompt, api_key, model, max_tokens, temperature, 
                base_url=None):
        """
        执行批量图片分析
        
        Args:
            images: 输入的图片张量列表
            prompt: 分析提示词
            api_key: OpenRoute API密钥
            model: 使用的模型名称
            max_tokens: 最大输出token数
            temperature: 生成温度
            base_url: API基础URL
            
        Returns:
            batch_results: 批量分析结果文本
        """
        # 使用配置文件中的默认值
        if base_url is None:
            base_url = OPENROUTE_CONFIG["base_url"]
            
        try:
            # 初始化OpenAI客户端
            client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            
            batch_size = images.shape[0]
            results = []
            
            print(f"🔄 开始批量处理 {batch_size} 张图片...")
            
            for i in range(batch_size):
                try:
                    print(f"📸 正在处理第 {i+1}/{batch_size} 张图片...")
                    
                    # 提取单张图片
                    single_image = images[i:i+1]
                    
                    # 转换为PIL图片
                    pil_image = self._tensor_to_pil(single_image)
                    
                    # 编码为base64
                    base64_image = self._encode_image_to_base64(pil_image)
                    
                    # 创建图片内容
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                    
                    # 创建消息
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                image_content
                            ]
                        }
                    ]
                    
                    # 调用API
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    # 获取分析结果
                    result = completion.choices[0].message.content
                    results.append(f"图片 {i+1}: {result}")
                    
                    print(f"✅ 第 {i+1} 张图片分析完成")
                    
                except Exception as e:
                    error_msg = f"图片 {i+1} 分析失败: {str(e)}"
                    results.append(error_msg)
                    print(f"❌ {error_msg}")
                    continue
            
            # 合并所有结果
            batch_results = "\n\n".join(results)
            print(f"🎉 批量处理完成，共处理 {batch_size} 张图片")
            
            return (batch_results,)
            
        except Exception as e:
            error_msg = f"批量图片分析失败: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return (error_msg,)
    
    def _tensor_to_pil(self, image_tensor):
        """将torch张量转换为PIL图片"""
        # 确保图片张量格式正确 (batch, height, width, channels)
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]  # 取第一个batch
        
        # 转换为numpy数组并确保值在0-255范围内
        if image_tensor.dtype == torch.float32:
            image_array = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_array = image_tensor.cpu().numpy().astype(np.uint8)
        
        # 创建PIL图片
        pil_image = Image.fromarray(image_array)
        return pil_image
    
    def _encode_image_to_base64(self, pil_image, quality=85):
        """将PIL图片编码为base64字符串"""
        # 确保图片是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为字节流
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
        img_byte_arr.seek(0)
        
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

# 节点映射
NODE_CLASS_MAPPINGS = {
    "OpenRouteImageAnalysis": OpenRouteImageAnalysis,
    "OpenRouteBatchImageAnalysis": OpenRouteBatchImageAnalysis,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouteImageAnalysis": "OpenRoute图片分析",
    "OpenRouteBatchImageAnalysis": "OpenRoute批量图片分析",
}
