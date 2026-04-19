# 应急知识文本处理
def split_text(text: str, chunk_size: int = 200):
    sentences = text.split("\n")
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + "\n"
        else:
            chunks.append(chunk.strip())
            chunk = sentence + "\n"

    if chunk:
        chunks.append(chunk.strip())
    return chunks