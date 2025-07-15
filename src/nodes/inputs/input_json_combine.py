import json
import copy
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


class InputJsonCombineV2:
    """
    输入的JSON字符串递归合并, 后面输入的JSON字符串会覆盖前面输入的JSON字符串

    输入：
        json输入_1: 输入的JSON字符串
        json输入_2: 输入的JSON字符串
        ...
    输出：
        combine_json: 递归合并后的JSON字符串
    """

    @classmethod
    def INPUT_TYPES(cls):
        json_dict = {
            f"json输入_{i+1}": (
                "STRING",
                {
                    "tooltip": f"输入的JSON字符串_{i+1}, 递归合并",
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

    _NODE_ZXQ = "JSON Combine V2 递归合并JSON"
    DESCRIPTION = "递归合并输入的JSON字符串，支持深层嵌套结构合并"
    CATEGORY = "ZXQ/合并json"

    def merge_json_obj(self, json_obj1, json_obj2):
        """递归合并两个JSON对象"""
        json_obj1 = copy.deepcopy(json_obj1)
        json_obj2 = copy.deepcopy(json_obj2)
        
        if isinstance(json_obj1, dict):
            for key, value in json_obj2.items():
                if key in json_obj1:
                    # 如果key在json_obj1中，则递归合并
                    json_obj1[key] = self.merge_json_obj(json_obj1[key], value)
                else:
                    # 如果key不在json_obj1中，则直接添加
                    json_obj1[key] = value
        elif isinstance(json_obj1, list):
            for i, item in enumerate(json_obj2):
                if i < len(json_obj1):
                    # 前n个元素，递归合并
                    json_obj1[i] = self.merge_json_obj(json_obj1[i], item)
                else:
                    # 如果json1的第i个元素不在json2中，则直接添加
                    json_obj1.append(item)
        else:
            # 如果json_obj1不是dict或list，则直接覆盖掉
            json_obj1 = json_obj2
        
        return json_obj1

    def forward(self, **kwargs):
        # 收集所有有效的JSON对象
        json_objects = []
        
        for json_name, json_str in kwargs.items():
            if json_str and str(json_str).strip():
                try:
                    tmp_json = json.loads(json_str)
                    json_objects.append(tmp_json)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，跳过这个输入
                    continue

        # 如果没有有效的JSON对象，返回空对象
        if not json_objects:
            return ("{}",)

        # 从第一个JSON对象开始，依次递归合并后续的对象
        result_json = json_objects[0]
        for json_obj in json_objects[1:]:
            result_json = self.merge_json_obj(result_json, json_obj)

        # 转换为JSON字符串输出
        output_json = json.dumps(result_json, indent=4, ensure_ascii=False)
        return (output_json,)
    

if __name__ == "__main__":
    print(InputJsonCombine.INPUT_TYPES())
    print(InputJsonCombineV2.INPUT_TYPES())
    # input_json_combine = InputJsonCombine()
    # input_json_combine.forward()
