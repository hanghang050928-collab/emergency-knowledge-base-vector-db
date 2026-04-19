from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import os
import requests

app = FastAPI()

# ======================
# 你自己的 KEY 粘贴在这里
# ======================
DASHSCOPE_API_KEY = "sk-4c1b1bc0918b45deb1ff2a28de8eea31"

# 向量数据库配置
CHROMA_PERSIST_DIRECTORY = "./vector_db_storage"
COLLECTION_NAME = "emergency_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 初始化向量库
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# 请求模型
class Query(BaseModel):
    question: str

class KnowledgeItem(BaseModel):
    title: str
    content: str

# 首页
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# --------------------------
# RAG + 通义千问官方AI
# --------------------------
@app.post("/api/ai_chat")
def ai_chat(query: Query):
    user_question = query.question.strip()
    if not user_question:
        return {"code": 400, "answer": "请输入问题"}

    # 1. 向量库检索
    res = collection.query(query_texts=[user_question], n_results=3)
    docs = res["documents"][0]

    if not docs:
        return {"code": 200, "answer": "知识库暂无相关内容，请先添加相关应急知识。"}

    context = "\n".join(docs)

    # 2. 构造提示词
    prompt = f"""
你是专业应急知识AI助手，请根据提供的知识，用自然、简洁、专业的语言回答。
只根据知识回答，不编造内容。

【相关知识】
{context}

【用户问题】
{user_question}

【AI回答】
"""

    # 3. 调用官方通义千问API
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"result_format": "message"}
    }

    try:
        resp = requests.post(url, json=data, headers=headers, timeout=20)
        answer = resp.json()["output"]["choices"][0]["message"]["content"]
        return {"code": 200, "answer": answer}
    except Exception as e:
        return {"code": 200, "answer": "AI暂时无法响应，请使用检索功能。"}

# --------------------------
# 普通检索（备用）
# --------------------------
@app.post("/api/search")
def search(query: Query):
    if not query.question.strip():
        return {"code":400, "data":[], "msg":"请输入问题"}
    res = collection.query(query_texts=[query.question], n_results=3)
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    result = []
    for i in range(len(documents)):
        score = round(100 - float(distances[i])*12, 2)
        if score < 50: continue
        result.append({"title": metadatas[i].get("title","应急知识"), "content": documents[i], "score": score})
    return {"code":200, "data":result}

# --------------------------
# 添加知识
# --------------------------
@app.post("/api/add")
def add_knowledge(item: KnowledgeItem):
    import uuid
    collection.add(
        documents=[item.content],
        metadatas=[{"title": item.title}],
        ids=[f"know_{uuid.uuid4().hex[:8]}"]
    )
    return {"code":200, "msg":"✅ 添加成功"}

# --------------------------
# 启动
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)