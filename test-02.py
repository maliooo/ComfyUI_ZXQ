from pymilvus import connections, Collection  # pip install pymilvus
import os
from src.nodes.nets.milvus_service import MilvusQuery

a = MilvusQuery()
b = a.query(
    query_type="text", 
    query_text="test", 
    collection_name="image_lora",
    search_field="desc_vector", top_k=1)
print(b)