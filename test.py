from src.utils.embedding import DoubaoEmbedding

doubao_embedding = DoubaoEmbedding()

res = doubao_embedding.get_text_embedding("你好")
print(res)
print(len(res))
print(type(res))
print(res.shape)