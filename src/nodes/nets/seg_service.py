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
import pandas as pd
from pathlib import Path
from torch import Tensor

ADE20K_CSV_PATH = Path(__file__).parent / "ade20k.csv"


class SegmentationService(BaseService):
    """
    通过调用OneFormer分割服务实现图像语义分割的ComfyUI节点。

    输入:
        image_base64: Base64编码的输入图像字符串。
        service_url: OneFormer分割服务的URL。
        resize_to: 图像预处理时调整大小的目标尺寸。
        alpha: 可视化结果时，分割掩码的透明度。
        show_text: 是否在可视化结果上显示标签文本。

    输出:
        visualized_image: 带有分割掩码的可视化图像 (IMAGE a.k.a. Tensor)。
        masks: 每个分割区域的掩码 (MASK a.k.a. Tensor)。
        labels_json: 包含标签和面积比例的JSON字符串。
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base64": ("STRING", ),
                "service_url": ("STRING", {"default": "http://172.31.48.6:28002/segment_image_base64"}),
                "resize_to": ("INT", {"default": 512, "min": 0, "max": 1536, "tooltip": "图片缩放，短边的长度，对于图片的完美像素"}),
                "show_text": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LIST", "LIST", "INT", "DICT", "STRING")
    RETURN_NAMES = ("visualized_image", "masks_image", "labels", "ratios", "count", "label_ratio_json", "label_ratio_json_str")
    FUNCTION = "segment_image"
    CATEGORY = "ZXQ/Segmentation"

    _NODE_NAME = "图像分割服务-segment"
    _NODE_ZXQ = "图像分割服务-segment"
    DESCRIPTION = "调用OneFormer图像分割服务"

    def segment_image(self, image_base64, service_url, resize_to, show_text):
        start_time = time.time()
        service_url = self.ip_map(service_url)  # 替换url
        if not image_base64:
            raise ValueError("image_base64 is required.")

        payload = {
            "image_url_or_base64": image_base64,
            "resize_to": resize_to,
            "show_text": show_text
        }

        try:
            response = requests.post(service_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            vis_image_b64 = result.get('vis_image')
            if not vis_image_b64:
                raise ValueError("Response does not contain 'vis_image'")
            
            image_data = base64.b64decode(vis_image_b64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            vis_image_tensor = torch.from_numpy(image_np)[None,]

            mask_images_b64 = result.get('mask_images', [])
            masks = []
            for mask_b64 in mask_images_b64:
                mask_data = base64.b64decode(mask_b64)
                mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
                mask_np = np.array(mask_image).astype(np.float32) / 255.0
                masks.append(torch.from_numpy(mask_np))
            
            if masks:
                masks_tensor = torch.stack(masks)
            else:
                h, w, _ = vis_image_tensor.shape[1:]
                masks_tensor = torch.zeros((0, h, w), dtype=torch.float32)

            end_time = time.time()
            print(f"自定义分割服务，分割时间: {(end_time - start_time):.2} seconds")
            assert len(result.get('label_info_list', [])) == len(result.get('areas_ratios', [])) == len(mask_images_b64), "label_info_list、areas_ratios、mask_images_b64长度不一致"

            return (
                vis_image_tensor, 
                masks_tensor, 
                result.get('label_info_list', []), 
                [float(item) for item in result.get('areas_ratios', [])], 
                len(result.get('label_info_list', [])),
                dict(zip(result.get('label_info_list', []), [float(item) for item in result.get('areas_ratios', [])])),
                json.dumps(dict(zip(result.get('label_info_list', []), [float(item) for item in result.get('areas_ratios', [])])), ensure_ascii=False)
            )

        except requests.exceptions.RequestException as e:
            error_message = f"API request failed: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)


class MasksProcessorService:
    """
    根据标签和面积比例合并和过滤掩码。

    输入:
        masks_image: 掩码张量 (IMAGE a.k.a. Tensor)。
        labels: 与每个掩码对应的标签列表 (STRING)。
        ratios: 与每个掩码对应的面积比例列表 (STRING)。
        need_label: 需要保留的标签，以逗号分隔 (STRING)。
        remove_label: 需要移除的标签，以逗号分隔 (STRING)。
        filter_ratio: 用于过滤掩码的最小面积比例 (FLOAT)。

    输出:
        merged_mask: 合并后的掩码 (MASK a.k.a. Tensor)。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks_image": ("IMAGE",),
                "labels_list": ("LIST", ),
                "ratios_list": ("LIST", ),
                "need_label": ("STRING", {"default": "", "tooltip": "需要保留的标签，用逗号分隔"}),
                "remove_label": ("STRING", {"default": "", "tooltip": "需要去除的标签，用逗号分隔"}),
                "filter_ratio": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "过滤掉小于比例的mask遮罩和标签"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "FLOAT")
    RETURN_NAMES = ("merged_mask_image", "merged_labels", "merged_ratio")
    FUNCTION = "process_masks"
    CATEGORY = "ZXQ/Segmentation"

    _NODE_NAME = "掩码处理器-segment_merger"
    _NODE_ZXQ = "掩码处理器-segment_merger"
    DESCRIPTION = "根据标签和面积比例合并和过滤掩码, 返回合并后的mask图片和合并后mask的比例"

    def _label_in_list(self, label, label_list):
        for index_item, label_item in enumerate(label_list):
            if str(label).strip().lower() in str(label_item).strip().lower():
                return index_item
        return -1

    def process_masks(self, masks_image, labels_list, ratios_list, need_label, remove_label, filter_ratio):

        if len(masks_image) != len(labels_list) or len(masks_image) != len(ratios_list):
            error_message = (f"assert长度错误，Inconsistent input lengths. "
                             f"Masks: {len(masks_image)}, "
                             f"Labels: {len(labels_list)}, "
                             f"Ratios: {len(ratios_list)}. "
                             "They must all be the same.")
            raise ValueError(error_message)

        # 处理字符串，将中文逗号替换为英文逗号
        if need_label:
            need_label = (need_label.strip()).replace("，", ",")
        if remove_label:
            remove_label = (remove_label.strip()).replace("，", ",")

        # 将字符串转换为集合，去除空格
        need_label_set = {label.strip(" ,，") for label in need_label.split(',') if label.strip()}
        remove_label_set = {label.strip(" ,，") for label in remove_label.split(',') if label.strip()}
        print(f"need_label_set: {need_label_set}")
        print(f"remove_label_set: {remove_label_set}")

        indices_to_keep = []
        
        # 先添加需要保留的掩码
        if need_label_set:
            for lebel_need in need_label_set:
                has_index = self._label_in_list(lebel_need, labels_list)  # 返回-1表示不存在
                if has_index != -1:
                    indices_to_keep.append(has_index)
        else:
            # 如果不需要保留，则添加所有掩码
            indices_to_keep = list(range(len(labels_list)))

        # 再移除需要移除的掩码
        for lebel_remove in remove_label_set:
            has_index = self._label_in_list(lebel_remove, labels_list)  # 返回-1表示不存在
            if has_index != -1:
                indices_to_keep.remove(has_index)
        
        # 移除面积比例小于过滤比例的掩码
        for index_item in indices_to_keep:
            if float(ratios_list[index_item]) < filter_ratio:
                indices_to_keep.remove(index_item)


        if not indices_to_keep:
            h, w = (masks_image.shape[1], masks_image.shape[2]) if len(masks_image) > 0 else (256, 256)
            merged_mask = torch.zeros((1, h, w), dtype=torch.float32)
            merged_ratio = 0 # 没有需要选择的，合并后的mask比例为0
            merged_labels = []
        else:
            filtered_masks = masks_image[indices_to_keep]
            merged_mask, _ = torch.max(filtered_masks, dim=0)
            merged_mask = merged_mask.unsqueeze(0)
            merged_ratio = sum([float(ratios_list[_index]) for _index in indices_to_keep])  # 合并所有选择的比例
            merged_labels = [labels_list[_index] for _index in indices_to_keep]

        
        # 确保合并后的掩码是4维的
        while merged_mask.dim() < 3:
            merged_mask = merged_mask.unsqueeze(0)
        if merged_mask.dim() == 3:
            merged_mask = (merged_mask.unsqueeze(-1)).repeat(1,1,1,3)

        return (merged_mask, merged_labels, merged_ratio)


class SegMergeJson:
    """
    根据图片分割结果和标签描述JSON，提取匹配的标签和详细描述。

    输入:
        image_json1: JSON字符串，key是label，value是对应key在图片中的比例。
        input_json2: JSON字符串，key是label，value是对应label的详细描述。
        ratio: 过滤比例，过滤掉小于ratio的key。

    输出:
        matched_labels: 匹配的标签列表。
        matched_descriptions: 对应的详细描述列表。
        matched_ratios: 对应的比例列表。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_json1": ("STRING", {
                    "tooltip": "图片分割结果JSON，key是label，value是比例",
                    "placeholder": "例子：{'wall':0.3, 'ceiling':0.2, 'floor':0.1, 'door':0.1, 'window':0.1}"
                }),
                "input_json2": ("STRING", {
                    "tooltip": "标签描述JSON，key是label，value是详细描述",
                    "placeholder": "例子：{'wall':'灰色乳胶漆墙面', 'ceiling':'黑色天花板', 'floor':'黑色大理石地面有着白色纹路', 'window':'玻璃窗，有着黑色边框,窗外城市风景'}"
                }),
                "keep_label_string": ("STRING", {
                    "default": "", 
                    "tooltip": "这里特指input_json2中需要保留的标签，用逗号分隔", 
                    "placeholder": "特指input_json2中需要保留的标签，即使图片里没有这个label也会保留。 例子：color,wall,ceiling,floor"
                }),
                "ratio": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "tooltip": "过滤掉小于比例的标签"}),
            }
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("matched_label_and_description",)
    FUNCTION = "merge_json"
    CATEGORY = "ZXQ/Segmentation"

    _NODE_NAME = "JSON合并器-segment_json_merger"
    _NODE_ZXQ = "JSON合并器-segment_json_merger"
    DESCRIPTION = "根据图片分割结果和标签描述JSON，提取匹配的标签和详细描述"

    def _label_in_list(self, label, label_list):
        """检查标签是否在列表中（不区分大小写）"""
        for index_item, label_item in enumerate(label_list):
            if str(label).strip().lower() in str(label_item).strip().lower():
                return index_item
        return -1

    def merge_json(self, image_json1, input_json2, keep_label_string, ratio):
        try:
            # 1. 确定image_json1，input_json2 是否是字符串
            image_json1 = self._info_to_json_str(image_json1)
            input_json2 = self._info_to_json_str(input_json2)

            # 解析JSON字符串
            image_data = json.loads(image_json1) if image_json1.strip() else {}
            input_data = json.loads(input_json2) if input_json2.strip() else {}
            
            # 过滤掉小于ratio的标签
            filtered_image_data = {k: v for k, v in image_data.items() if float(v) >= ratio}
            # 处理keep_label
            if keep_label_string:
                keep_label_string = (keep_label_string.strip()).replace("，", ",")
                keep_label_dict = {label.strip(" ,，"): 1 for label in keep_label_string.split(',') if label.strip()}
                # 并集
                keep_label_dict.update(filtered_image_data)
                filtered_image_data = keep_label_dict
                
            
            matched_label_and_description = {}
            
            # 遍历过滤后的图片数据
            for image_label, _ in filtered_image_data.items():
                # 在input_json2中查找匹配的标签
                for input_label, input_description in input_data.items():
                    # 不区分大小写匹配
                    if (str(image_label).strip().lower() in str(input_label).strip().lower()) or (str(input_label).strip().lower() in str(image_label).strip().lower()):
                        matched_label_and_description[image_label] = input_description
                        break  # 找到第一个匹配就跳出内层循环
            
            return (json.dumps(matched_label_and_description, ensure_ascii=False), )
            
        except json.JSONDecodeError as e:
            error_message = f"JSON解析错误: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        except Exception as e:
            error_message = f"处理过程中发生错误: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        
    def _info_to_json_str(self, info):
        """将输入输入转换为json格式字符串"""
        if isinstance(info, dict):
            return json.dumps(info, ensure_ascii=False)
        elif isinstance(info, list):
            return json.dumps(info, ensure_ascii=False)
        elif isinstance(info, str):
            return info
        else:
            return json.dumps(info, ensure_ascii=False)


class SegLabelRatioSelector:
    """
    根据用户选择的标签，从label_ratio_json_str中提取对应的比例值。

    输入:
        label_ratio_json_str: JSON字符串，包含标签和对应的比例值。
        selected_label: 用户选择的标签名称。

    输出:
        selected_ratio: 选中标签对应的比例值。
        found: 是否找到对应的标签。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 读取ADE20K的csv文件
        ade20k_pandas_df = pd.read_csv(ADE20K_CSV_PATH)
        
        return {
            "required": {
                "label_ratio_json_str": ("STRING", {
                    "tooltip": "包含标签和比例值的JSON字符串",
                    "placeholder": "例子：{'wall':0.3, 'ceiling':0.2, 'floor':0.1, 'door':0.1, 'window':0.1}"
                }),
                "selected_label": (ade20k_pandas_df["Name"].tolist(), {
                    "tooltip": "用户选择的标签名称",
                    "placeholder": "例如：wall"
                }),
            },
            "optional": {
                "merge_label_ratio_str": ("STRING", {
                    "tooltip": "串连合并的标签比例JSON字符串",
                    "placeholder": "例子：{'wall':0.3, 'ceiling':0.2}"
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("selected_ratio", "found", "selected_label_ratio_str", "input_label_ratio_str")
    FUNCTION = "get_ratio_by_label"
    CATEGORY = "ZXQ/Segmentation"

    _NODE_NAME = "标签比例选择器-segment_label_ratio_selector_ADE20K"
    _NODE_ZXQ = "标签比例选择器-segment_label_ratio_selector_ADE20K"
    DESCRIPTION = "根据用户选择的标签，从JSON字符串中提取对应的比例值"

    def _label_in_dict(self, target_label, label_dict):
        """检查标签是否在字典中（不区分大小写）"""
        target_label = str(target_label).strip().lower()
        for label in label_dict.keys():
            if target_label in str(label).strip().lower() or str(label).strip().lower() in target_label:
                return label
        return None

    def get_ratio_by_label(self, label_ratio_json_str, selected_label, merge_label_ratio_str = None):
        try:

            # 确保输入是字符串格式
            label_ratio_json_str = self._info_to_json_str(label_ratio_json_str)
            
            # 解析JSON字符串
            label_ratio_data = json.loads(label_ratio_json_str) if label_ratio_json_str.strip() else {}
            
            if not selected_label or not selected_label.strip():
                return (0.0, False)
            
            # 查找匹配的标签
            matched_label = self._label_in_dict(selected_label, label_ratio_data)
            found = True if matched_label is not None else False  # 是否找到对应的标签
            selected_ratio = 0.0 if not found else float(label_ratio_data[matched_label])

            # 如果selected_label_ratio_str不为空，则合并
            selected_label_ratio_data = json.loads(merge_label_ratio_str) if merge_label_ratio_str.strip() else {}


            if matched_label is not None:
                selected_label_ratio_data[matched_label] = label_ratio_data[matched_label]
            
            return (selected_ratio, found, json.dumps(selected_label_ratio_data, ensure_ascii=False), label_ratio_json_str)
            
                
        except json.JSONDecodeError as e:
            error_message = f"JSON解析错误: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        except Exception as e:
            error_message = f"处理过程中发生错误: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        
    def _info_to_json_str(self, info):
        """将输入转换为json格式字符串"""
        if isinstance(info, dict):
            return json.dumps(info, ensure_ascii=False)
        elif isinstance(info, list):
            return json.dumps(info, ensure_ascii=False)
        elif isinstance(info, str):
            return info
        else:
            return json.dumps(info, ensure_ascii=False)

class MaskedNearestFill:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "ZXQ/Segmentation"
    FUNCTION = "fill"

    _NODE_NAME = "最近邻填充-segment_nearest_fill"
    _NODE_ZXQ = "最近邻填充-segment_nearest_fill"
    DESCRIPTION = "使用边界众数颜色填充mask区域"

    def fill(self, image: Tensor, mask: Tensor):
        import cv2
        import numpy as np
        
        image = image.detach().clone()
        mask = self._mask_unsqueeze(self._mask_floor(mask))
        assert mask.shape[0] == image.shape[0], "Image and mask batch size does not match"
        
        for slice_idx, (image_slice, mask_slice) in enumerate(zip(image, mask)):
            # 转换为numpy数组，并确保正确的格式
            image_np = (image_slice.cpu().numpy() * 255).astype(np.uint8)
            mask_np = (mask_slice.squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            # 确保图像是BGR格式（OpenCV格式）
            if image_np.shape[2] == 3:
                # 如果是RGB格式，转换为BGR
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # 创建二值mask
            _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
            output_image = image_np.copy()
            
            # 找到所有轮廓
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 处理每个轮廓
            for contour in contours:
                # 创建临时mask用于当前轮廓
                temp_mask = np.zeros_like(mask_binary)
                cv2.drawContours(temp_mask, [contour], -1, 255, thickness=cv2.FILLED)
                
                # 创建边界带
                kernel = np.ones((5, 5), np.uint8)
                dilated_mask = cv2.dilate(temp_mask, kernel, iterations=1)
                border_strip_mask = cv2.subtract(dilated_mask, temp_mask)
                
                # 提取边界带内的像素
                border_pixels = output_image[border_strip_mask == 255]
                
                if border_pixels.size > 0:
                    # 找到众数颜色
                    unique_colors, counts = np.unique(border_pixels, axis=0, return_counts=True)
                    mode_color_bgr = unique_colors[counts.argmax()]
                    fill_color = tuple(int(c) for c in mode_color_bgr)
                else:
                    # 备用方案：使用黑色
                    fill_color = (0, 0, 0)
                
                # 填充当前轮廓区域
                cv2.drawContours(output_image, [contour], -1, fill_color, thickness=cv2.FILLED)
            
            # 转换回RGB格式并归一化
            if output_image.shape[2] == 3:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            
            # 转换回tensor格式
            output_tensor = torch.from_numpy(output_image.astype(np.float32) / 255.0)
            image[slice_idx] = output_tensor
        
        return (image,)
    
    def _mask_unsqueeze(self, mask: Tensor):
        if len(mask.shape) == 3:  # BHW -> B1HW
            mask = mask.unsqueeze(1)
        elif len(mask.shape) == 2:  # HW -> B1HW
            mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def _mask_floor(self, mask: Tensor, threshold: float = 0.5):
        return (mask >= threshold).to(mask.dtype)



NODE_CLASS_MAPPINGS = {
    "SegmentationService": SegmentationService,
    "MasksProcessorService": MasksProcessorService,
    "SegMergeJson": SegMergeJson,
    "SegLabelRatioSelector": SegLabelRatioSelector,
    "MaskedNearestFill": MaskedNearestFill
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegmentationService": "图像分割服务",
    "MasksProcessorService": "掩码处理器",
    "SegMergeJson": "SEG-颜色材质-JSON合并器",
    "SegLabelRatioSelector": "SEG-标签比例选择器",
    "MaskedNearestFill": "最近邻填充-分割图"
}
