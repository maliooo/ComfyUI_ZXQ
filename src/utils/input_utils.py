import json
import requests
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal


input_info_json_cache = {
    "value": None,
    "last_update": None,
}


def get_input_keys():
    """
    获取输入参数的键
    """
    return list(get_input_json_info().keys())


def get_input_json_info():  # -> Any:
    """
    获取输入参数的键
    """
    global input_info_json_cache
    cache_value = input_info_json_cache["value"]
    cache_time = input_info_json_cache["last_update"]
    if cache_value is not None and cache_time is not None and (datetime.now() - cache_time) < timedelta(hours=1):
        print("从缓存中获取input_info_json模板")
        content = cache_value
    else:
        cache_file = Path(__file__).parents[2] / "input_template.json"
        json_url = "https://cdn.code.znzmo.com/aiPainting/ai_workflow_json_exp.json"
        try:
            print("从CDN获取input_info_json模板")
            response = requests.get(json_url, timeout=3)
            response.raise_for_status()
            response.encoding = "utf-8"
            content = response.text
            input_info_json_cache["value"] = content
            input_info_json_cache["last_update"] = datetime.now()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(content, encoding="utf-8")
        except Exception as e:
            print(
                f"从CDN {json_url} 获取输入ai_workflow_json_exp.json失败从获取输入参数的键失败，尝试从本地缓存文件 {cache_file} 获取: {e}\n{traceback.format_exc()}"
            )
            if cache_file.exists():
                content = cache_file.read_text(encoding="utf-8")
            else:
                raise ValueError(f"从{json_url}获取输入ai_workflow_json_exp.json失败从获取输入参数的键失败，且本地没有缓存文件: {e}\n{traceback.format_exc()}")
    return json.loads(content)

def value_to_type(value:str, type:str = Literal["INT", "FLOAT", "STRING", "BOOLEAN", "LIST", "DICT"]):
    """
    将字符串转换为指定类型
    """
    if value == "" or str(value).strip() == "":
        return None

    if type == "INT":
        return int(value)
    elif type == "FLOAT":
        return float(value)
    elif type == "STRING":
        return str(value)
    elif type == "BOOLEAN":
        return bool(value)
    elif type == "LIST":
        return json.loads(value)
    elif type == "DICT":
        return json.loads(value)
    else:
        print(f"[red]输入{value}, 不支持的类型: {type}[/red]")
        return None


if __name__ == "__main__":
    print(get_input_keys())
    print(get_input_json_info())
