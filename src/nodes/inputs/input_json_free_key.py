import json
from ...utils import value_to_type

NODE_NUM = 5

class InputJsonFreeKey:
    """
    Input JSON free key, 自由输入的JSON字符串

    输出：
        text: 自由组合的JSON字符串
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dict = {}
        type_list = ["INT", "FLOAT", "STRING", "BOOLEAN", "LIST", "DICT"]
        for i in range(NODE_NUM):
            input_dict[f"json_key_{i}"] = (
                "STRING",
                {
                    "default": "",
                }
            )
            input_dict[f"json_type_{i}"] = (
                type_list,
                {
                    "default": type_list[2],
                }
            )

            input_dict[f"json_value_{i}"] = (
                "STRING",
                {
                    "default": "",
                }
            )

        return {
            "optional": {**input_dict}
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_json",)
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_NAME = "Input JSON free key"
    _NODE_ZXQ = "自由输入key的JSON字符串"
    DESCRIPTION = "自由输入的JSON字符串"
    CATEGORY = "ZXQ/自由输入key的JSON字符串"

    def forward(self, **kwargs):
        output_json = {}
        for i in range(NODE_NUM):
            key = kwargs[f"json_key_{i}"]
            value = kwargs[f"json_value_{i}"]
            type = kwargs[f"json_type_{i}"]
            
            if key:
                if key != "" and str(key).strip() != "":
                    if value:
                        if value != "" and str(value).strip() != "":
                            # 将value转换为type类型
                            value_type = value_to_type(value, type)
                            if value_type:
                                output_json[key] = value_type
                else:
                    continue
        output_json_str = json.dumps(output_json, indent=4, ensure_ascii=False)

        return (output_json_str,)
            