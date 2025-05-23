import json
import requests
import traceback
from pathlib import Path
from typing import Literal
from rich import print

def get_input_keys():
    """
    获取输入参数的键
    """
    return list(get_input_json_info().keys())


def get_input_json_info():  # -> Any:
    """
    获取输入参数的键
    """
    cache_file = Path(__file__).parents[2] / "input_template.json"
    json_url = "https://cdn.code.znzmo.com/aiPainting/ai_workflow_json_exp.json"
    try:
        response = requests.get(json_url, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"
        content = response.text
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(content, encoding="utf-8")
    except Exception as e:
        print(
            f"从{json_url}获取输入ai_workflow_json_exp.json失败从获取输入参数的键失败: {e}\n{traceback.format_exc()}"
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
