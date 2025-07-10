import os
from pathlib import Path
import json

class BaseService:
    """
    基础服务类
    """
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def ip_map(self, url):
         # 1. 读取配置文件，替换ip地址
        florence_json_con_path = Path(__file__).parent / "config.json"
        if os.path.exists(florence_json_con_path):
            with open(florence_json_con_path, "r") as f:
                config = json.load(f)
            florence_ip_map = config.get("florence_ip_map", {})
            if florence_ip_map:
                for source_ip, target_ip in florence_ip_map.items():
                    if source_ip in url:
                        url = url.replace(source_ip, target_ip)
                        print(f"[yellow]调用ip_map函数，替换ip地址: {source_ip} -> {target_ip}[/yellow]")

        return url

            