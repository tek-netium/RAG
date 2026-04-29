# RAG 知识库问答系统

基于 LangChain 和 Ollama 的本地 RAG（Retrieval-Augmented Generation）知识库问答系统。上传 PDF 或 Markdown 文档，系统自动解析、切分、向量化并建立索引，用户通过自然语言提问即可获得基于文档内容的精准回答。

## 核心特性

- **文档处理**：支持 PDF（含 OCR 公式识别）和 Markdown，通过 Docling 转换为结构化文本
- **混合检索**：密集向量检索（BAAI/bge-m3）+ BM25 关键词检索，Ensemble 加权融合，提升召回质量
- **本地 LLM**：通过 Ollama 运行本地模型（如 Qwen），全程离线，数据安全
- **Agent 模式**：LangChain Agent 自动调用检索工具，支持流式输出和来源引用
- **API 服务**：FastAPI + SSE 流式响应，支持文档上传/删除/列表管理，热更新检索器无需重启
- **React 前端**：现代化聊天界面，会话管理，文档管理面板
- **对话记忆**：Redis 存储多轮对话历史，超时自动清理
- **质量评估**：内置 RAGAS 评估脚本，支持 faithfulness 和 answer_relevancy 指标

## 技术栈

| 层级 | 技术 |
|------|------|
| 文档解析 | Docling + EasyOCR |
| 文本切分 | LangChain RecursiveCharacterTextSplitter |
| 向量存储 | Chroma |
| 嵌入模型 | BAAI/bge-m3 (HuggingFace) |
| 混合检索 | Dense Vector + BM25 + EnsembleRetriever |
| LLM | Ollama (qwen3.5:9b) |
| Agent | LangChain Agent + Tool |
| API | FastAPI + SSE |
| 前端 | React + Vite |
| 对话记忆 | Redis |
| 评估 | RAGAS |

## 项目结构

- core/                          核心模块
  - agent.py                     Agent 构建与流式输出
  - retriever.py                 混合检索器与检索工具
  - document_processor.py        文档发现、去重与入库
  - vectorstore.py               Chroma 向量库构建
  - text_splitter.py             文本递归切分
  - loaders.py                   文档加载（PDF/MD）
  - pdf_to_markdown.py           PDF 转 Markdown（OCR）
  - ingestion_state.py           文件处理状态记录
  - app_factory.py               RAG 应用组装工厂
- api.py                         FastAPI + SSE 流式 API
- main.py                        CLI 交互入口
- evaluate.py                    RAGAS 评估脚本
- config.py                      配置数据类
- models.py                      LLM/Embedding 构建
- rag-chat-frontend/             React + Vite 前端
- requirements.txt               Python 依赖
- .gitignore

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Redis（对话记忆）
redis-server

# 3. 启动 Ollama 并拉取模型
ollama pull qwen3.5:9b

# 4. 启动 API 服务
python api.py

# 5. （可选）启动前端
cd rag-chat-frontend && npm install && npm run dev

# 6. CLI 模式
python main.py your_document.pdf
```

## 评估

```bash
# 配置 DEEPSEEK_API_KEY 环境变量后运行
python evaluate.py
```
