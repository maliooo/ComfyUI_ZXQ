import pandas as pd
import os

class SavePandasZnzmo:
    """
    Save Pandas DataFrame to Local
    将Pandas DataFrame保存到本地文件

    输入：
        df: Pandas DataFrame数据
        save_path: 保存路径
    输出：
        success: 是否保存成功
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "df": ("DATAFRAME",),
                "save_path": ("STRING", {
                    "default": "",
                    "tooltip": "保存文件的完整路径，例如：/path/to/save/data.csv"
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("success",)
    OUTPUT_NODE = True

    FUNCTION = "forward"

    _NODE_NAME = "Save Pandas znzmo 保存数据"
    _NODE_ZXQ = "保存Pandas数据到本地"
    DESCRIPTION = "将Pandas DataFrame保存到本地文件，支持csv、excel等格式"
    CATEGORY = "ZXQ/数据保存"

    def forward(self, df, save_path):
        try:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 根据文件扩展名选择保存方式
            file_ext = os.path.splitext(save_path)[1].lower()
            
            if file_ext == '.csv':
                df.to_csv(save_path, index=False, encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                df.to_excel(save_path, index=False)
            elif file_ext == '.json':
                df.to_json(save_path, orient='records', force_ascii=False, indent=4)
            else:
                # 默认保存为csv
                df.to_csv(save_path, index=False, encoding='utf-8')
            
            return (True,)
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
            return (False,) 