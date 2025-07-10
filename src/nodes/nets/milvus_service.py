import json
from pymilvus import connections, Collection  # pip install pymilvus


import numpy as np
from PIL import Image
import io
import base64
from ...utils import DoubaoEmbedding, get_znzmo_embedding
import time
import os
import omegaconf
from pathlib import Path

class MilvusQuery:
    """
    Milvus向量数据库查询节点

    输入：
        query_type: 查询类型 (text/image)
        query_content: 查询内容 (文本或base64编码的图片)
        collection_name: Milvus集合名称
        top_k: 返回最相似的k个结果
    输出：
        search_results: 查询结果JSON字符串
    """
    def __init__(self):
        self.config = omegaconf.OmegaConf.load(Path(__file__).parents[3] / "config" / "llm.yaml")
        self.milvus_config = self.config.MILVUS
        # self.connections = connections.connect(
        #     uri = self.milvus_config.uri,
        #     token = self.milvus_config.token
        # )
        # print(f"连接Milvus成功, 使用uri: {self.milvus_config.uri}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query_type": (
                    ["text", "image"], 
                    {
                        "default": "text",
                        "tooltip": "选择查询类型, 文本或图像",
                    }
                ),
                "query_text": (
                    "STRING", 
                    {
                        "tooltip": "输入的查询文本",
                    }
                ),
                "query_image_base64": (
                    "STRING", 
                    {
                        "tooltip": "输入的查询图像base64编码",
                    }
                ),
                "collection_name": (
                    "STRING", 
                    {
                        "default": "image_lora",
                        "tooltip": "输入Milvus集合名称",
                    }
                ),
                "search_field": (
                    ["desc_vector", "image_vector"], 
                    {
                        "default": "desc_vector",
                        "tooltip": "选择搜索字段, 描述向量或图像向量",
                    }
                ),
                "filter_expr": (
                    "STRING", 
                    {
                        "tooltip": "过滤条件",
                    }
                ),
                "top_k": (
                    "INT", 
                    {
                        "default": 1, 
                        "min": 1, 
                        "max": 100,
                        "tooltip": "返回最相似的k个结果",
                    }
                ),
            },
            "optional": {
                "host": (
                    "STRING", 
                    {
                    }
                ),
                "port": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("json_list", "key_names")
    OUTPUT_NODE = False

    _NODE_NAME = "Milvus 向量数据库查询"
    _NODE_ZXQ = "Milvus 向量数据库查询"
    DESCRIPTION = "Milvus 向量数据库查询"

    FUNCTION = "query"
    CATEGORY = "ZXQ/Milvus"

    def query(
        self, 
        query_type = None, 
        query_text = None, 
        query_image_base64 = None, 
        collection_name = None, 
        search_field = None, 
        filter_expr = None,
        top_k = None, 
        host = None, 
        port = None
    ):
        try:
            # 连接Milvus
            connections.connect(
                uri = self.milvus_config.uri,
                token = self.milvus_config.token
            )
            collection = Collection(collection_name)
            collection.load()
            print(f"连接Milvus成功, 使用uri: {self.milvus_config.uri}")
            print(f"加载了collection: {collection.name}")
            # print(f"连接Milvus成功: {collection_name}")
            # collection.load()

            if query_text is None and query_image_base64 is None:
                raise ValueError("query_text 和 query_image_base64 不能同时为空")
            
            if query_type == "text" and query_text is None:
                raise ValueError("当query_type为text时, query_text不能为空")
            if query_type == "image" and query_image_base64 is None:
                raise ValueError("当query_type为image时, query_image_base64不能为空")

            # 根据查询类型处理输入
            if query_type == "text":
                # 这里需要根据实际情况实现文本到向量的转换
                query_vector = self._text_to_vector(query_text)
            else:  # image
                # 处理base64编码的图片
                image_data = base64.b64decode(query_image_base64)
                image = Image.open(io.BytesIO(image_data))
                # 这里需要根据实际情况实现图像到向量的转换
                query_vector = self._image_to_vector(image)

            # 执行向量搜索
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field = search_field,
                param=search_params,
                limit=top_k,
                output_fields=["*"],
                expr=filter_expr
            )

            # 处理搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    info_dict = {}
                    for key, value in hit.fields.items():
                        if not isinstance(value, list):
                            info_dict[key] = value
                    search_results.append(json.dumps(info_dict, ensure_ascii=False))

            return (
                search_results, 
                json.loads(search_results[0]).keys()
            )

        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
        finally:
            connections.disconnect("default")

    def _text_to_vector(self, text):
        """
        将文本转换为向量, 使用豆包模型
        """
        # TODO: 实现文本到向量的转换
        doubao_embedding = DoubaoEmbedding()
        return doubao_embedding.get_text_embedding(text)

    def _image_to_vector(self, base64_image):
        """
        将图像转换为向量, 使用知末模型
        """
        # TODO: 实现图像到向量的转换
        # 将base64转换为pil图片
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        # 保存到临时文件
        tmp_dir = "/home/public/filebrowser/data/测试-01/临时文件"
        temp_file = f"{tmp_dir}/temp_{time.time()}.jpg"
        image.save(temp_file)
        # 使用get_znzmo_embedding函数将图片转换为向量
        image_vector = get_znzmo_embedding(temp_file, "image")
        # 删除临时文件
        os.remove(temp_file)

        return image_vector

if __name__ == "__main__":
    a = MilvusQuery()
    b = a.query(
        query_type="text", 
        query_text="test", 
        collection_name="image_lora",
        search_field="desc_vector", top_k=1)
    print(b)
    print(len(b))
    print(type(b[0][0]))
    cc = b[0][0]
    for key, value in json.loads(cc).items():
        print(key, value)
    # print(np.random.rand(2048).shape)