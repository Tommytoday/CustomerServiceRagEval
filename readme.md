以下是根据我们整个项目实战过程，整理出的**完整、清晰、适合直接使用的 `README.md`**：

---

# 葆婴公司 AI 智能客服与产品知识 RAG 系统评测项目

## 项目简介

本项目实现了一个**完全本地化**的母婴营养补充品知识 RAG（Retrieval-Augmented Generation）系统，专为葆婴（USANA）类保健食品场景设计。

系统特点：
- 使用本地 LLM（Ollama + Qwen2.5）
- 向量数据库：Qdrant（支持 metadata 过滤）
- 评测框架：DeepEval（Faithfulness + Answer Relevancy）
- 严格合规 Prompt（保健食品不得夸大功效、必须提醒咨询医生）
- 完全离线运行，适合企业内部评测与 PoC

## 项目结构

```
baoying-rag-eval/
├── docs/                    # 模拟产品知识文档（TXT）
├── qdrant_storage/          # Qdrant 本地持久化数据
├── rag_evl_Drant.py         # 主程序（推荐文件名）
├── venv/                    # Python 虚拟环境
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 1. 安装 Ollama 并拉取模型
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b          # 用于评测 judge
ollama pull nomic-embed-text    # Embedding 模型

# 2. 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -U langchain langchain-community langchain-ollama langchain-qdrant qdrant-client deepeval
```

### 2. 准备文档

将模拟产品知识文档放入 `docs/` 文件夹（推荐 8-12 个独立主题文档）：
- `haizao_you.txt` → 海藻油DHA胶囊
- `gai_jia_D.txt` → 钙加D片
- `duowei_yingyang_pian_yunfu.txt` → 孕妇多维营养片
- `tuihuo_guize.txt` → 退货规则
- `anquan_shenming.txt` → 安全声明
- `ye_suan_bu_chong.txt` → 叶酸补充
- `peihe_jianyi.txt` → 产品搭配建议
- `baozhiqi_cunchu.txt` → 保质期与储存
- ...（更多文档可自行扩展）

每个文档需包含清晰的产品名称、成分、适用人群、服用建议、注意事项等。

### 3. 运行系统

```bash
python rag_evl_Drant.py
```

系统会：
- 自动重建 Qdrant 向量库（REBUILD_VECTORSTORE = True 时）
- 为每个文档添加 metadata（product 字段）
- 使用 metadata filter 精确检索
- 输出单条测试回答
- 运行完整的 10 条测试集 DeepEval 评测

## 核心配置说明

### Prompt 设计原则
- 严格 grounding（只使用上下文事实）
- 强制添加保健食品 disclaimer
- 禁止医疗诊断与功效夸大
- 要求使用完整产品名称

### Qdrant 配置
- 本地持久化存储（`./qdrant_storage`）
- 支持 metadata filter（按产品名称精确检索）
- 维度：nomic-embed-text（通常 768）

### 评测指标
- **Faithfulness**：回答是否忠实于检索到的上下文（防幻觉）
- **Answer Relevancy**：回答是否直接相关用户问题

## 常见问题与优化

| 问题                     | 解决方案                              |
|--------------------------|---------------------------------------|
| 检索噪声大               | 添加/完善 metadata + 使用 Filter      |
| 回答不够精准             | 加强 Prompt + temperature=0.0         |
| 产品名称不完整           | Prompt 中强调“使用完整产品名称”      |
| Qdrant 关闭警告          | 可忽略，或在脚本末尾添加 client.close() |
| 评测超时                 | 使用 qwen2.5:3b 作为 judge + async_mode=False |

## 后续扩展方向

- 多轮对话记忆（LangGraph）
- Agent 工具调用（查订单、库存）
- 多模态支持（产品图片理解）
- 生产部署（Docker + FastAPI + 前端聊天界面）
- 定期知识库更新机制

## 项目成果

通过本项目，你可以：
- 验证 RAG 在垂直母婴保健品领域的实际效果
- 量化评估检索质量与生成忠实度
- 掌握本地化企业级知识库构建与评测全流程
- 为真实葆婴智能客服系统提供技术验证与优化基础

---

**作者**：基于 Grok 实战指导构建  
**日期**：2026 年  
**用途**：学习、演示、内部 PoC

---

你可以直接把上面内容保存为 `README.md` 文件。

需要我再补充以下任意部分吗？
- 项目目录结构更详细说明
- 依赖版本锁定（requirements.txt）
- 如何运行完整评测的详细步骤
- 未来优化路线图（带优先级）

直接告诉我你的需求，我马上帮你补充或调整 README 内容。 

这个 README 已经可以直接用于项目交付或内部分享了。