import chromadb
from chromadb.utils import embedding_functions
from code.config import *

# 初始化向量数据库客户端
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

# 嵌入函数（文本 → 向量）
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# 获取/创建集合
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"description": "应急管理知识库"}
)

# 添加文档到向量库
def add_documents(documents: list, ids: list, metadatas: list = None):
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

# 语义检索
def search(query: str, n_results: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

# 查看所有数据
def get_all():
    return collection.get()