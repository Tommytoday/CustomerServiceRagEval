# ====================== rag_eval.py ======================
# 葆婴 RAG 本地评测版 - 支持 Qdrant 向量数据库（推荐替换 Chroma）

import os
os.environ["OPENAI_API_KEY"] = "fake-key"   # 让 Ragas/DeepEval 不报错

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 新增 Qdrant 支持
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ====================== 配置 ======================
REBUILD_VECTORSTORE = True                    # 改文档后设为 True
DOCS_DIR = "./docs"
QDRANT_COLLECTION = "baoying_products"        # 集合名称
QDRANT_PATH = "./storage"              # 本地持久化路径（推荐）

LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"              # 或 "mxbai-embed-large"

# ====================== 1. 加载文档 ======================
loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
# 在 splits = text_splitter.split_documents(docs) 之后添加：
#添加metadata
for doc in splits:
    content = doc.page_content
    if "海藻油DHA胶囊" in content:
        doc.metadata["product"] = "海藻油DHA胶囊"
    elif "钙加D片" in content:
        doc.metadata["product"] = "钙加D片"
    elif "多维营养片" in content:
        doc.metadata["product"] = "多维营养片"
    elif "退货" in content or "退款" in content:
        doc.metadata["product"] = "退货规则"
    elif "保质期" in content or "储存" in content:
        doc.metadata["product"] = "保质期储存"
    else:
        doc.metadata["product"] = "general"

print(f"✅ 加载并切分完成，共 {len(splits)} 个文档块")

# ====================== 2. Embedding + Qdrant ======================
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# 使用本地磁盘持久化模式（推荐，不依赖服务器）
client = QdrantClient(path=QDRANT_PATH)   # 本地路径持久化

if REBUILD_VECTORSTORE:
    try:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        print(f"已删除旧集合: {QDRANT_COLLECTION}")
    except Exception:
        pass

# 创建集合（如果不存在）
if not client.collection_exists(QDRANT_COLLECTION):
    # nomic-embed-text 维度一般是 768，确认你的 embedding 维度
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"创建新集合: {QDRANT_COLLECTION}")

# 从文档构建向量存储
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

# 如果需要重建（插入文档）
if REBUILD_VECTORSTORE:
    vectorstore.add_documents(splits)
    print(f"✅ 已将 {len(splits)} 个文档块存入 Qdrant")
from qdrant_client.http import models as qdrant_models   # 在文件顶部添加
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3 ,# 先用5条，减少噪声
        #"filter": {"product": "葆婴海藻油DHA胶囊"}   # 只检索海藻油相关文档
        "filter": qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="product",
                    match=qdrant_models.MatchValue(value="葆婴海藻油DHA胶囊")
                )
            ]
        )
    }
)

print(f"✅ Qdrant 向量库构建完成！集合: {QDRANT_COLLECTION}，文档块: {len(splits)}")

# ====================== 3. LLM + 严格 Prompt ======================
llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

template = """你是葆婴官方AI智能客服助手。必须绝对严格、逐字遵守以下规则：

1. 严格使用上下文中的**完整产品名称**和**原话**回答，不能简化、不能改写、不能添加任何词语。
2. 对于剂量和服用方法，必须**完全复制**上下文中的文字，例如“孕妇每日1-2粒，随餐服用”。
3. **禁止**使用“根据现有产品信息无法确定”这句话，除非上下文里真的没有任何相关内容。
4. 每条回答**最后必须**精确加上这一句：“本品为保健食品，不能代替药物。请咨询医生。”
5. 回答必须简洁、专业、友好，使用中文。

上下文：
{context}

问题：{question}

答案：
"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ====================== 测试 & 评测部分（保持原有） ======================
if __name__ == "__main__":
    print("\n🔍 测试单个查询：")
    test_query = "孕妇可以吃海藻油胶囊吗？每天吃几粒？"
    print("Q:", test_query)

    retrieved_docs = retriever.invoke(test_query)
    print("\n--- 检索到的前3条上下文 ---")
    for i, doc in enumerate(retrieved_docs[:3]):
        print(f"[{i + 1}] {doc.page_content[:400]}...\n")

    answer = rag_chain.invoke(test_query)
    print("A:", answer)
    # 优雅关闭 Qdrant 客户端，减少警告
    try:
        if 'client' in locals():
            client.close()
    except:
        pass