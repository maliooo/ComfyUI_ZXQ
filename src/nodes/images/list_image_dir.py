import os
import random
import glob
from pathlib import Path

DEFAULT_FOLDER_PATH = r""

class ListImageDir:
    """
    列出输入文件夹地址中所有文件夹，然后可以根据seed读取指定文件夹图片

    输入：
        folder_path: 输入的文件夹路径
        seed: 随机种子，用于选择指定文件夹中的图片
        image_extensions: 图片文件扩展名，默认为 jpg,jpeg,png,gif,bmp,webp
    输出：
        image: 根据seed选择的文件夹中的图片
        image_path: 根据seed选择的文件夹中的图片路径
        image_list: 选中文件夹中的所有图片路径列表
        image_count: 选中文件夹中的图片总数
    """

    @classmethod
    def INPUT_TYPES(cls):
        folder_path_list = os.listdir(DEFAULT_FOLDER_PATH)
        folder_path_list = [folder_path for folder_path in folder_path_list if os.path.isdir(os.path.join(DEFAULT_FOLDER_PATH, folder_path))]
        return {
            "required": {
                "folder_path": (
                    folder_path_list,
                    {
                        "default": folder_path_list[0],
                        "tooltip": f"根据默认{DEFAULT_FOLDER_PATH}文件夹路径，将列出该路径下的所有子文件夹",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999999,
                        "tooltip": "随机种子，用于选择指定文件夹中的图片",
                    },
                ),
                "image_extensions": (
                    "STRING",
                    {
                        "default": "jpg,jpeg,png,gif,bmp,webp",
                        "tooltip": "图片文件扩展名，用逗号分隔",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("selected_folder", "image_list", "folder_count")
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_ZXQ = "List Image Dir 列出文件夹图片"
    DESCRIPTION = "列出输入文件夹地址中所有文件夹，然后可以根据seed读取指定文件夹图片"
    CATEGORY = "ZXQ/图片处理"

    def forward(self, folder_path, seed, image_extensions):
        # 检查文件夹路径是否存在
        if not folder_path or not os.path.exists(folder_path):
            return ("", "", 0)
        
        # 获取所有子文件夹
        subfolders = []
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    subfolders.append(item_path)
        except Exception as e:
            print(f"读取文件夹时出错: {e}")
            return ("", "", 0)
        
        folder_count = len(subfolders)
        if folder_count == 0:
            return ("", "", 0)
        
        # 使用seed选择文件夹
        random.seed(seed)
        selected_folder = random.choice(subfolders)
        
        # 获取图片扩展名列表
        extensions = [ext.strip().lower() for ext in image_extensions.split(",")]
        
        # 获取选中文件夹中的所有图片
        image_files = []
        try:
            for ext in extensions:
                pattern = os.path.join(selected_folder, f"*.{ext}")
                image_files.extend(glob.glob(pattern))
                # 也查找大写扩展名
                pattern = os.path.join(selected_folder, f"*.{ext.upper()}")
                image_files.extend(glob.glob(pattern))
        except Exception as e:
            print(f"读取图片文件时出错: {e}")
            return (selected_folder, "", folder_count)
        
        # 将图片路径列表转换为字符串
        image_list = "\n".join(image_files)
        
        return (selected_folder, image_list, folder_count)


if __name__ == "__main__":
    print(ListImageDir.INPUT_TYPES())
    # list_image_dir = ListImageDir()
    # result = list_image_dir.forward("/path/to/folder", 123, "jpg,png")
    # print(result)
