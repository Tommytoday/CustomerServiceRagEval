# ====================== rag_eval.py ======================
# 葆婴 RAG 本地评测完整版（修复 DeepEval + Ragas 兼容问题）

import os
os.environ["OPENAI_API_KEY"] = "fake-key"   # 让 Ragas 不报错

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ====================== 配置 ======================
REBUILD_VECTORSTORE = True
DOCS_DIR = "./docs"
CHROMA_DB_DIR = "./chroma_db"
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

# ====================== 1. 加载文档 & 构建向量库 ======================
if REBUILD_VECTORSTORE and os.path.exists(CHROMA_DB_DIR):
    import shutil
    shutil.rmtree(CHROMA_DB_DIR)

loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6, "filter": None})  # 可后续加 metadata filter)

print(f"✅ 向量库构建完成！共 {len(splits)} 个文档块")

# ====================== 2. LLM + Prompt ======================
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

template = """你是葆婴官方AI智能客服助手。必须严格遵守以下规则：

1. 只使用提供的上下文中的事实回答，**禁止编造任何信息**。
2. 如果上下文没有明确答案，回复：“根据现有产品信息无法确定，建议咨询医生或葆婴官方客服。”
3. 禁止医疗诊断、治疗、功效夸大。必须加上：“本品为保健食品，不能代替药物。”
4. 回答要简洁、专业、友好，使用中文。

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

# ====================== 3. 测试单个查询 ======================
if __name__ == "__main__":
    print("\n🔍 测试单个查询：")
    test_query = "孕妇可以吃海藻油胶囊吗？每天吃几粒？"
    print("Q:", test_query)
    print("A:", rag_chain.invoke(test_query))

    # ====================== 4. 测试数据集 ======================
    questions = [
        "孕妇可以吃海藻油胶囊吗？每天吃几粒？",
        "钙加D片和多维营养片可以一起服用吗？",
        "海藻油DHA的主要作用是什么？",
        "产品保质期是多久？如何储存？",
        "3岁宝宝能吃多维营养片吗？",
        "这个产品能治疗近视吗？",
        "退货需要什么条件？",
        "海藻油和钙加D片可以同时吃吗？",
        "孕期每天需要补充多少叶酸？",
        "产品过敏了怎么办？"
    ]

    ground_truths = [
        "孕妇每日1-2粒，随餐服用。本品为保健食品，请咨询医生。",
        "一般可以搭配使用，但建议查看产品说明或咨询医生。",
        "DHA支持胎儿/婴幼儿大脑和眼睛发育，是神经系统重要成分。",
        "保质期24个月，储存于阴凉干燥处，避免阳光直射。",
        "3岁以上儿童可按说明服用，具体请咨询儿科医生。",
        "本品为保健食品，不能代替药物治疗任何疾病。请咨询医生。",
        "未开封且在保质期内可退货，需提供完整包装和发票。",
        "可以同时服用，但建议间隔30分钟或咨询医生。",
        "孕期叶酸补充建议遵医嘱，本品含叶酸但不能替代处方药。",
        "立即停止服用并咨询医生，必要时就医。"
    ]

    # 生成答案和上下文
    answers = []
    contexts = []
    for q in questions:
        retrieved_docs = retriever.invoke(q)
        contexts.append([doc.page_content for doc in retrieved_docs])
        answers.append(rag_chain.invoke(q))

    # ====================== 5. DeepEval 评测（修复版） ======================
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
    from deepeval.models import OllamaModel   # ← 关键导入

    print("\n🚀 开始 DeepEval 评测（使用 OllamaModel）...")

    # 创建 DeepEval 支持的 OllamaModel（温度设低，判断更稳定）
    judge_model = OllamaModel(
        model=LLM_MODEL,           # "qwen2.5:7b"
        base_url="http://localhost:11434",
        temperature=0.0
    )

    test_cases = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        test_case = LLMTestCase(
            input=q,
            actual_output=a,
            expected_output=gt,
            retrieval_context=ctx
        )
        test_cases.append(test_case)

    faithfulness_metric = FaithfulnessMetric(model=judge_model, threshold=0.7)
    answer_relevancy_metric = AnswerRelevancyMetric(model=judge_model, threshold=0.7)
    contextual_precision_metric = ContextualPrecisionMetric(model=judge_model, threshold=0.7)

    deepeval_result = deepeval_evaluate(
        test_cases=test_cases,
        metrics=[faithfulness_metric, answer_relevancy_metric, contextual_precision_metric]
    )
    print("✅ DeepEval 评测完成！")

    # ====================== 5. DeepEval 评测（最终防超时简化版） ======================
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.models import OllamaModel

    print("\n🚀 开始 DeepEval 评测（最终简化版）...")

    # 使用更小的 judge 模型（推荐 qwen2.5:3b，速度快很多）
    # 如果你还没下载：ollama pull qwen2.5:3b
    judge_model = OllamaModel(
        model="qwen2.5:3b",  # ← 改成 3b 显著降低超时风险
        base_url="http://localhost:11434",
        temperature=0.0
    )

    # 只保留两个最核心指标，并关闭每个 metric 的异步
    faithfulness_metric = FaithfulnessMetric(
        model=judge_model,
        threshold=0.7,
        async_mode=False  # 关键：关闭异步
    )

    answer_relevancy_metric = AnswerRelevancyMetric(
        model=judge_model,
        threshold=0.7,
        async_mode=False  # 关键：关闭异步
    )

    test_cases = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        test_case = LLMTestCase(
            input=q,
            actual_output=a,
            expected_output=gt,
            retrieval_context=ctx
        )
        test_cases.append(test_case)

    # 调用 evaluate 时不要传 run_async
    deepeval_result = deepeval_evaluate(
        test_cases=test_cases,
        metrics=[faithfulness_metric, answer_relevancy_metric]
    )
    print("✅ DeepEval 评测完成！")