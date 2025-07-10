import requests
import numpy as np
import json
from openai import OpenAI
from .llm_config import LLM_CONFIG

def get_znzmo_embedding(url_or_text: str, method:str):
    """
    获取知末的embedding

    Args:
        url_or_text: 文本或图片URL
        method: 方法, 可选值为 "text" 或 "image"
    
    Returns:
        embedding: 嵌入向量, 图片是640维, 文本是1024维
        
    """

    import requests
    url = "http://14.103.44.55:8001/tovec/"
    params = {
        "url_or_text": url_or_text,
        "method": method
    }
    response_img = requests.get(url, params=params, timeout=2)
    print(len(response_img.json()))  # 图片的编码长度为640维度
    return np.array(response_img.json())



class DoubaoEmbedding:
    """豆包嵌入模型封装类
    
    用于调用豆包嵌入模型API，将文本转换为向量表示。
    """
    
    def __init__(self, api_key: str = LLM_CONFIG.MYSELF.ARK_API_KEY):
        """初始化豆包嵌入模型客户端
        
        Args:
            api_key (str): API密钥，默认为示例密钥
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.model = "doubao-embedding-large-text-250515"
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """获取单个文本的嵌入向量
        
        Args:
            text (str): 输入文本
            
        Returns:
            np.ndarray: 文本的嵌入向量
        """
        resp = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float"
        )
        
        json_resp = json.loads(resp.model_dump_json())
        json_resp_data = json_resp["data"]
        embedding = json_resp_data[0]["embedding"]
        return np.array(embedding)
    
    def get_text_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """获取多个文本的嵌入向量
        
        Args:
            texts (list[str]): 输入文本列表
            
        Returns:
            list[np.ndarray]: 文本嵌入向量列表
        """
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        
        json_resp = json.loads(resp.model_dump_json())
        json_resp_data = json_resp["data"]
        embeddings = [np.array(item["embedding"]) for item in json_resp_data]
        return embeddings


if __name__ == "__main__":
    res = get_znzmo_embedding("https://image2.znzmo.com/1608871217061_944.jpeg", "image")
    # print(res)
    print(len(res))
    print(type(res))
    print(res.shape)