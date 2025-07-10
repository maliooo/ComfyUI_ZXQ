import json
NUM_INPUT_JSON = 5

class InputJsonCombine:
    """
    输入的JSON字符串合并

    输入：
        json输入_1: 输入的JSON字符串
        json输入_2: 输入的JSON字符串
    输出：
        combine_json: 合并后的JSON字符串, 以最前面的输入参数为准
    """

    @classmethod
    def INPUT_TYPES(cls):
        json_dict = {
            f"json输入_{i+1}": (
                "STRING",
                {
                    "tooltip": f"输入的JSON字符串_{i+1}, 以最前面的输入参数为准",
                },
            )
            for i in range(NUM_INPUT_JSON)
        }

        return {
            "optional": {**json_dict},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combine_json",)
    OUTPUT_NODE = False

    FUNCTION = "forward"

    _NODE_ZXQ = "JSON Combine 输入的JSON字符串合并"
    DESCRIPTION = "合并输入的JSON字符串，以最前面的输入参数为准"
    CATEGORY = "ZXQ/合并json"

    def forward(self, **kwargs):
        dict_json = {}
        for json_name, json_str in kwargs.items():
            if json_str:
                if str(json_str).strip() == "":
                    continue
                try:
                    tmp_json = json.loads(json_str)
                except:
                    continue

                for key, value in tmp_json.items():
                    if key not in dict_json:
                        dict_json[key] = value
                    else:
                        pass # 如果key已经存在，则不进行合并

        output_json = json.dumps(dict_json, indent=4, ensure_ascii=False)
        return (output_json,)
    

if __name__ == "__main__":
    print(InputJsonCombine.INPUT_TYPES())
    # input_json_combine = InputJsonCombine()
    # input_json_combine.forward()
