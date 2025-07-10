import pandas as pd
import os

class SavePandasData:
    """
    Save Pandas DataFrame
    将pandas DataFrame保存到本地文件

    输入：
        df: pandas DataFrame数据
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

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "save_path")
    OUTPUT_NODE = False

    FUNCTION = "save_data"

    _NODE_NAME = "Save Pandas Data"
    _NODE_ZXQ = "保存pandas数据"
    DESCRIPTION = "将pandas DataFrame保存到本地文件"
    CATEGORY = "ZXQ/数据处理"

    def save_data(self, df, save_path):
        try:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 根据文件扩展名选择保存格式
            file_ext = os.path.splitext(save_path)[1].lower()
            if file_ext == '.csv':
                df.to_csv(save_path, index=False)
            elif file_ext == '.xlsx':
                df.to_excel(save_path, index=False)
            elif file_ext == '.json':
                df.to_json(save_path, orient='records', force_ascii=False)
            elif file_ext == '.parquet':
                df.to_parquet(save_path, index=False)
            else:
                # 默认保存为CSV
                df.to_csv(save_path, index=False)

            return (True, save_path)
        except Exception as e:
            print(f"保存数据时出错: {str(e)}")
            return (False, "")
