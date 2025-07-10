import os
import random
import glob
from pathlib import Path
from ...utils.image_utils import load_image_by_path
from rich import print

# try:
#     import debugpy
#     print("[yellow]debugpy is loaded[/yellow]")
#     debugpy.listen(5678)
#     print("[yellow]debugpy is listening on port 5678[/yellow]")
#     debugpy.wait_for_client()
#     print("[yellow]debugpy is connected[/yellow]")
# except ImportError:
#     print("[red]debugpy is not installed[/red]")

DEFAULT_FOLDER_PATH = r"/Volumes/Lexar-4T/Lexar-下载/方案深化渲染-测试图集"  # 默认文件夹路径
FILEBROWSER_URL_PREFIX = "http://1.180.12.34:58100/api/public/dl/YNjtgXCT"  # filebrowser的url前缀
FILEBROWSER_DIR_PREFIX = "/home/public/filebrowser/data/测试-01"  # filebrowser的url替换文件夹路径前缀，用于将服务器上的文件路径转换为filebrowser的url

class ListImageDir:
    """
    列出输入文件夹地址中所有文件夹，然后可以根据seed读取指定文件夹图片

    输入：
        folder_path: 输入的文件夹路径
        seed: 随机种子，用于选择指定文件夹中的图片
        image_extensions: 图片文件扩展名，默认为 jpg,jpeg,png,gif,bmp,webp
    输出：
        image: 根据seed选择的文件夹中的图片
        mask: image对应的图片的mask
        image_path: 根据seed选择的文件夹中的图片路径
        image_count: 选中文件夹中的图片总数
    """

    @classmethod
    def INPUT_TYPES(cls):
        folder_path_list = os.listdir(DEFAULT_FOLDER_PATH)
        folder_path_list = [folder_path for folder_path in folder_path_list if os.path.isdir(os.path.join(DEFAULT_FOLDER_PATH, folder_path))]
        folder_path_list = ["None"] + folder_path_list
        return {
            "required": {
                "folder_path": (
                    folder_path_list,
                    {
                        "default": folder_path_list[0],
                        "tooltip": f"根据默认【{DEFAULT_FOLDER_PATH}】文件夹路径，将列出该路径下的所有子文件夹",
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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "image_path", "filebrowser_url", "image_count")
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_ZXQ = "List Image Dir 列出文件夹图片"
    DESCRIPTION = "列出输入文件夹地址中所有文件夹，然后可以根据seed读取指定文件夹图片"
    CATEGORY = "ZXQ/图片处理"

    def forward(self, folder_path, seed, image_extensions):
        # 检查文件夹路径是否存在
        if not folder_path or not os.path.exists(Path(DEFAULT_FOLDER_PATH) / folder_path) or folder_path == "None":
            return (None, None, None, None, 0)

        
        # 获取文件夹下的所有图片文件, 并过滤掉非图片文件, 并返回图片路径列表, 并排序
        image_files = os.listdir(Path(DEFAULT_FOLDER_PATH) / folder_path)
        image_files = [file for file in image_files if file.endswith(tuple(image_extensions.split(","))) and os.path.isfile(os.path.join(DEFAULT_FOLDER_PATH, folder_path, file))]
        image_files = [os.path.join(DEFAULT_FOLDER_PATH, folder_path, file) for file in image_files]
        image_files = [file for file in image_files if os.path.isfile(file)]
        image_files.sort()

        # 根据seed选择图片
        selected_image_path = image_files[seed % len(image_files)]
        selected_image = load_image_by_path(selected_image_path)
        image, mask = selected_image

        # filebrowser_url, 将selected_image_path转换为filebrowser_url
        filebrowser_url = None
        if FILEBROWSER_DIR_PREFIX in selected_image_path:
            # 将selected_image_path中的FILEBROWSER_DIR_PREFIX替换为FILEBROWSER_URL_PREFIX，将服务器上的文件路径转换为filebrowser的url
            filebrowser_url = selected_image_path.replace(FILEBROWSER_DIR_PREFIX, FILEBROWSER_URL_PREFIX)

        return (image, mask, selected_image_path, filebrowser_url, len(image_files))
        


if __name__ == "__main__":
    print(ListImageDir.INPUT_TYPES())
    # list_image_dir = ListImageDir()
    # result = list_image_dir.forward("/path/to/folder", 123, "jpg,png")
    # print(result)
