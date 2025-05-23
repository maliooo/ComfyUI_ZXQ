import json
from ...utils import get_input_json_info

class InputJsonInfoZnzmo:
    """
    Input Info JSON string
    提供Input Info JSON模板

    输入：
        text: 输入的JSON字符串
    输出：
        text: 输入的JSON字符串
    """

    @classmethod
    def INPUT_TYPES(cls):
        znzmo_input_json_info = get_input_json_info()
        input_dict = {
        }

        for key, value in znzmo_input_json_info.items():
            if isinstance(value, int):
                input_dict[key] = (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": f"输入{key}的值, 如果是-1, 则表示不使用这个输入",
                    }
                )
            elif isinstance(value, float):
                input_dict[key] = (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "tooltip": f"输入{key}的值, 如果是-1.0, 则表示不使用这个输入",
                    }
                )
            elif isinstance(value, str):
                input_dict[key] = (
                    "STRING",
                    {
                        "tooltip": f"输入{key}的值, 如果是空字符串, 则表示不使用这个输入",
                    }
                )
            elif isinstance(value, bool):
                input_dict[key] = (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": f"输入{key}的值, 如果是False, 则表示不使用这个输入",
                    }
                )
            elif isinstance(value, list):
                input_dict[key] = (
                    "LIST",
                    {
                        "default": [],
                        "tooltip": f"输入{key}的值, 如果是空列表, 则表示不使用这个输入",
                    }
                )
            elif isinstance(value, dict):
                input_dict[key] = (
                    "DICT",
                    {
                        "default": {},
                        "tooltip": f"输入{key}的值, 如果是空字典, 则表示不使用这个输入",
                    }
                )
            else:
                raise ValueError(f"不支持的类型: {type(value)}")

        return {
            "optional": {**input_dict},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_json",)
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_NAME = "Input JSON znzmo 知末格式字符串"
    _NODE_ZXQ = "输入的JSON字符串"
    DESCRIPTION = "根据znzmo的输入信息, 生成知末格式JSON字符串"
    CATEGORY = "ZXQ/知末格式JSON"

    def forward(self, **kwargs):
        output_json = {}
        for key, value in kwargs.items():
            if isinstance(value, int):
                if value != -1:
                    output_json[key] = value
            elif isinstance(value, float):
                if value >= 0:
                    output_json[key] = value
            elif isinstance(value, str):
                if value != "" and str(value).strip() != "":
                    output_json[key] = value
            elif isinstance(value, bool):
                if value:
                    output_json[key] = value
            elif isinstance(value, list):
                if value != []:
                    output_json[key] = value
            elif isinstance(value, dict):
                if value != {}:
                    output_json[key] = value
            else:
                raise ValueError(f"不支持的类型: {type(value)}")
                
        return (json.dumps(output_json, indent=4, ensure_ascii=False),)
