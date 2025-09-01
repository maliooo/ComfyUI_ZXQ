import os
import importlib
import inspect
from rich import print  # pip install rich

__all__ = []
# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 遍历当前目录下的所有.py文件
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        # 获取模块名（不包含.py后缀）
        module_name = filename[:-3]
        
        # 动态导入模块
        module = importlib.import_module(f'.{module_name}', package=__package__)  # package=__package__ 表示当前包, 当前包是comfyui_zxq
        
        # 遍历模块中的所有成员
        for name, obj in inspect.getmembers(module):
            # 检查是否是类且具有__NODE_ZXQ属性
            if isinstance(obj, type) and hasattr(obj, '_NODE_ZXQ'):
                # 将类名（字符串）添加到__all__列表
                __all__.append(name)
                # 将类对象导入到当前命名空间
                globals()[name] = obj
                # print(f"[red]已经加载： {name}[/red]")


