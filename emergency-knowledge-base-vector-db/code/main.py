from code.vector_db.connection import add_documents, search
from code.knowledge_base.parser import split_text

# ----------------------
# 步骤1：构造应急管理知识库
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
# 步骤2：把知识存入向量数据库
# ----------------------
documents = []
metadatas = []
ids = []

for idx, item in enumerate(emergency_data):
    documents.append(item["content"])
    metadatas.append({"title": item["title"]})
    ids.append(f"emergency_{idx}")

add_documents(documents=documents, metadatas=metadatas, ids=ids)
print("✅ 应急知识已成功存入向量数据库！")

# ----------------------
# 步骤3：测试语义检索（核心功能）
# ----------------------
print("\n===== 测试检索 =====")
question = "有人触电应该怎么办？"
results = search(question)

print("问题：", question)
print("\n最相关的应急知识：")
for i, doc in enumerate(results["documents"][0]):
    print(f"\n【结果{i+1}】")
    print(doc)
