from langchain_community.embeddings import DashScopeEmbeddings

user_input = "猫坐在垫子上"

# 创建嵌入模型对象，默认使用text-embedding-v1
embedding = DashScopeEmbeddings()
result = embedding.embed_query(user_input)
# <class 'list'>
print(type(result))
# 1536
print(len(result))
print(result)
