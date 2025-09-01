from rich import print  # pip install rich
from .nodes import *
import inspect

print("这是一个测试")

NODE_CLASS_MAPPINGS = {
}

# Optionally, you can rename the node in the `NODE_DISPLAY_NAME_MAPPINGS` dictionary.
NODE_DISPLAY_NAME_MAPPINGS = {
}

# 导入src中的所有obj， 判断是否是class，如果是class，则导入
for name, obj in list(globals().items()):
    # print(f"[blue]name: {name}, obj: {obj}[/blue]")
    if isinstance(obj, type) and hasattr(obj, "_NODE_ZXQ"):
        node_name = f"ZXQ_{name}"
        NODE_CLASS_MAPPINGS[node_name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = f"{obj._NODE_ZXQ} (ZXQ NODES)"

print(f"[yellow]已经加载： ZXQ NODES LOADED, 一共有{len(NODE_CLASS_MAPPINGS)}个节点[/yellow]")

# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]