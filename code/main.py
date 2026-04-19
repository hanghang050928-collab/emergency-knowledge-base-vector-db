import chromadb
from chromadb.utils import embedding_functions

# ----------------------
# 配置
# ----------------------
CHROMA_PERSIST_DIRECTORY = "./vector_db_storage"
COLLECTION_NAME = "emergency_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ----------------------
# 初始化向量数据库
# ----------------------
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"description": "应急管理知识库"}
)

# ----------------------
# 应急知识库数据
# ----------------------
emergency_data = [
    {
        "title": "火灾逃生",
        "content": "发生火灾时应低姿前进，用湿毛巾捂口鼻，切勿乘坐电梯，沿安全出口迅速撤离。"
    },
    {
        "title": "地震避险",
        "content": "地震发生时立即躲在桌子下或墙角，护住头部，远离玻璃窗和吊灯，停止晃动后再撤离。"
    },
    {
        "title": "溺水急救",
        "content": "发现溺水者先呼救，将漂浮物递给溺水者，上岸后清理口鼻异物，必要时进行心肺复苏。"
    },
    {
        "title": "触电处理",
        "content": "有人触电立即切断电源，用干燥木棍分离电源与伤者，切勿直接用手接触伤者。"
    }
]

# ----------------------
# 存入向量库
# ----------------------
documents = []
metadatas = []
ids = []

for idx, item in enumerate(emergency_data):
    documents.append(item["content"])
    metadatas.append({"title": item["title"]})
    ids.append(f"emergency_{idx}")

collection.add(documents=documents, metadatas=metadatas, ids=ids)
print("✅ 应急知识已成功存入向量数据库！")

# ----------------------
# 测试检索
# ----------------------
print("\n===== 测试检索 =====")
question = "有人触电应该怎么办？"
results = collection.query(query_texts=[question], n_results=3)

print("问题：", question)
print("\n最相关的应急知识：")
for i, doc in enumerate(results["documents"][0]):
    print(f"\n【结果{i+1}】")
    print(doc)