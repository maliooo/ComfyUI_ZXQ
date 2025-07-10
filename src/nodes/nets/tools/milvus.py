from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker
)
import numpy as np
from rich import print
import omegaconf
from pathlib import Path
config = omegaconf.OmegaConf.load(Path(__file__).parents[4] / "config" / "llm.yaml")


class MilvusDB:
    def __init__(self, host=config.MILVUS.host, port=config.MILVUS.port, token=config.MILVUS.token, uri=config.MILVUS.uri):
        """初始化 Milvus 连接"""
        self.host = host
        self.port = port 
        self.token = token
        self.uri = uri
        self.connect()
        
    def connect(self):
        """连接到 Milvus 服务器"""
        if self.token is None:
            connections.connect(host=self.host, port=self.port)
            print(f"[green]连接Milvus成功, 使用host方式： {self.host}:{self.port}[/green]")
        else:
            connections.connect(uri=self.uri, token=self.token)
            print(f"[green]连接Milvus成功, 使用uri方式： {self.uri}[/green]")
        
    def create_collection(self, collection_name):
        """创建集合"""
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"[red]删除集合: {collection_name}[/red]")
    
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="desc_vector", dtype=DataType.FLOAT_VECTOR, dim=2048),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=640),
            FieldSchema(name="style", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="qwen_max_prompt", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=4096),
        ]
        
        schema = CollectionSchema(fields=fields, description="Image LoRA collection")
        collection = Collection(name=collection_name, schema=schema)
        print(f"[green]创建集合: {collection_name}[/green]")
        print(f"[green]创建字段: {fields}[/green]")
        print(f"[green]创建schema: {schema}[/green]")
        print("-"*50)

        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="desc_vector", index_params=index_params)
        collection.create_index(field_name="image_vector", index_params=index_params)
        print(f"[green]创建索引: {collection_name}[/green]")
        
        return collection
    
    def insert_data(self, collection_name, desc_vectors, image_vectors, styles, descs, qwen_max_prompt, image_urls,):
        """插入数据到集合
        
        Args:
            desc_vectors: 描述向量列表，每个向量维度为2048
            image_vectors: 图像向量列表，每个向量维度为640
            styles: 风格列表
            descs: 描述列表
            qwen_max_prompt: 提示词列表
        """
        collection = Collection(collection_name)
        
        entities = [
            desc_vectors,
            image_vectors,
            styles,
            descs,
            qwen_max_prompt,
            image_urls
        ]
        
        collection.insert(entities)
        collection.flush()
        print(f"[green]插入数据: {self.collection_name}, 数量: {len(desc_vectors)}[/green]")
        print("-"*50)
        
    def search_by_desc_vector(self, collection_name, query_vector, top_k=10, filter=None):
        """通过描述普通的向量搜索，普通搜索（search）：先进行向量相似度搜索得到topk个结果，然后再进行过滤
        
        Args:
            query_vector: 查询向量，维度为2048
            top_k: 返回最相似的k个结果
            filter: 过滤条件，例如 'style in ["奶油风"]'
            
        Returns:
            搜索结果列表
        """
        collection = Collection(collection_name)
        collection.load()
        
        # 确保查询向量是一维的且是浮点数类型
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.flatten().tolist()
        elif isinstance(query_vector, list) and isinstance(query_vector[0], list):
            query_vector = query_vector[0]
            
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # 如果提供了过滤条件，确保它是字符串类型
        if filter and not isinstance(filter, str):
            filter = str(filter)
            
        # 如果使用的是 search是先出topk个结果，然后过滤， 如果使用的是 hybrid search， 是先过滤，然后出topk个结果
        # 使用 hybrid search， 这里使用混合搜索

        results = collection.search(
            data=[query_vector],
            anns_field="desc_vector",
            param=search_params,
            limit=top_k,
            output_fields=["style", "desc", "qwen_max_prompt", "image_url"],
            expr=filter
        )
        
        return results
    
    def search_by_image_vector(self, collection_name, query_vector, top_k=10, filter=None):
        """通过图像普通向量搜索，普通搜索（search）：先进行向量相似度搜索得到topk个结果，然后再进行过滤
        
        Args:
            query_vector: 查询向量，维度为640
            top_k: 返回最相似的k个结果
            filter: 过滤条件，例如 'style in ["奶油风"]'
            
        Returns:
            搜索结果列表
        """
        collection = Collection(collection_name)
        collection.load()
        
        # 确保查询向量是一维的且是浮点数类型
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.flatten().tolist()
        elif isinstance(query_vector, list) and isinstance(query_vector[0], list):
            query_vector = query_vector[0]
            
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # 如果提供了过滤条件，确保它是字符串类型
        if filter and not isinstance(filter, str):
            filter = str(filter)
            

        results = collection.search(
            data=[query_vector],
            anns_field="image_vector",
            param=search_params,
            limit=top_k,
            output_fields=["style", "desc", "qwen_max_prompt", "image_url"],
            expr=filter
        )
        
        return results
    
    def hybrid_search_by_desc_vector(self, collection_name, query_vector, top_k=10, filter=None):
        """通过描述向量进行混合搜索，混合搜索（hybrid_search）：先进行过滤，然后再进行向量相似度搜索得到topk个结果
        
        Args:
            query_vector: 查询向量，维度为2048
            top_k: 返回最相似的k个结果
            filter: 过滤条件，例如 'style in ["奶油风"]'
            
        Returns:
            搜索结果列表
        """
        collection = Collection(collection_name)
        collection.load()
        
        # 确保查询向量是一维的且是浮点数类型
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.flatten().tolist()
        elif isinstance(query_vector, list) and isinstance(query_vector[0], list):
            query_vector = query_vector[0]
            
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # 如果提供了过滤条件，确保它是字符串类型
        if filter and not isinstance(filter, str):
            filter = str(filter)
            
        # 构建搜索请求
        search_requests = [
            AnnSearchRequest(
                data=[query_vector],
                anns_field="desc_vector",
                param=search_params,
                limit=top_k,
                expr=filter
            )
        ]
        
        # 设置重排序参数
        rerank_params = {
            "strategy": "weighted",
            "weights": [1.0]
        }
        
        # 执行混合搜索
        results = collection.hybrid_search(
            reqs=search_requests,
            rerank=WeightedRanker(*([1.0]*len(search_requests))),
            limit=top_k,
            output_fields=["style", "desc", "qwen_max_prompt", "image_url"]
        )
        
        return results
    
    def hybrid_search_by_image_vector(self, collection_name, query_vector, top_k=10, filter=None):
        """通过图像向量进行混合搜索，混合搜索（hybrid_search）：先进行过滤，然后再进行向量相似度搜索得到topk个结果
        
        Args:
            query_vector: 查询向量，维度为640
            top_k: 返回最相似的k个结果
            filter: 过滤条件，例如 'style in ["奶油风"]'
            
        Returns:
            搜索结果列表
        """
        collection = Collection(collection_name)
        collection.load()
        
        # 确保查询向量是一维的且是浮点数类型
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.flatten().tolist()
        elif isinstance(query_vector, list) and isinstance(query_vector[0], list):
            query_vector = query_vector[0]
            
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # 如果提供了过滤条件，确保它是字符串类型
        if filter and not isinstance(filter, str):
            filter = str(filter)
            
        # 构建搜索请求
        search_requests = [
            AnnSearchRequest(
                data=[query_vector],
                anns_field="image_vector",
                param=search_params,
                limit=top_k,
                expr=filter
            )
        ]
        
        # 设置重排序参数
        rerank_params = {
            "strategy": "weighted",
            "weights": [1.0]
        }
        
        # 执行混合搜索
        results = collection.hybrid_search(
            reqs=search_requests,
            rerank=WeightedRanker(*([1.0]*len(search_requests))),
            limit=top_k,
            output_fields=["style", "desc", "qwen_max_prompt", "image_url"]
        )
        
        return results
    
    def close(self):
        """关闭连接"""
        connections.disconnect("default")


if __name__ == "__main__":
    milvus = MilvusDB()
    # milvus.create_collection()
    # milvus.close()

    # 创建一批 fake数据插入
    # desc_vectors = np.random.rand(100, 2048)
    # image_vectors = np.random.rand(100, 640)
    # styles = []
    # descs = []
    # prompts = []
    # for i in range(len(desc_vectors)):
    #     styles.append(f"style{i}")
    #     descs.append(f"desc{i}")
    #     prompts.append(f"prompt{i}")
        
    # milvus.insert_data(desc_vectors, image_vectors, styles, descs, prompts)
    # milvus.close()

    # 搜索示例
    query_vector = np.random.rand(2048).tolist()  # 创建一维向量
    results = milvus.search_by_desc_vector(
        collection_name="image_lora", 
        query_vector=query_vector, 
        top_k=1,
        filter = "style in ['新中式', '奶油风']"
        )
    print("搜索结果：")
    for hits in results:
        for hit in hits:
            print(f"ID: {hit.id}, 距离: {hit.distance}")
            print(f"Style: {hit.entity.get('style')}")
            print(f"Desc: {hit.entity.get('desc')}")
            print(f"Prompt: {hit.entity.get('qwen_max_prompt')}")
            print("-" * 50)
    
    milvus.close()