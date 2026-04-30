# RAG 知识库问答系统

基于 LangChain 和 Ollama 的本地 RAG（Retrieval-Augmented Generation）知识库问答系统。上传 PDF 或 Markdown 文档，系统自动解析、切分、向量化并建立索引，用户通过自然语言提问即可获得基于文档内容的精准回答。

## 核心特性

- **文档处理**：支持 PDF（含 OCR 公式识别）和 Markdown，通过 Docling 转换为结构化文本，支持**单文件和整个文件夹**上传
- **混合检索**：密集向量检索（BAAI/bge-m3）+ BM25 关键词检索，Ensemble 加权融合，按会话隔离
- **本地 LLM**：通过 Ollama 运行本地模型（Qwen 3.5 4B），全程离线，数据安全
- **Agent 模式**：LangChain Agent 自动调用检索工具，支持流式输出和来源引用
- **多会话管理**：创建/切换/删除独立会话，每个会话拥有独立的文档空间和对话历史
- **API 服务**：FastAPI + SSE 流式响应，文档上传/删除/列表管理，会话 CRUD，CORS 跨域支持
- **React 前端**：现代化聊天界面，会话侧边栏，文档管理面板，思考计时显示，上传进度条
- **对话记忆**：Redis 存储多轮对话历史，超时自动清理
- **质量评估**：内置 RAGAS 评估脚本，支持 faithfulness 和 answer_relevancy 指标
- **环境变量驱动**：关键配置通过环境变量覆盖，方便本地开发与云部署切换

## 技术栈

| 层级 | 技术 |
|------|------|
| 文档解析 | Docling + EasyOCR |
| 文本切分 | LangChain RecursiveCharacterTextSplitter |
| 向量存储 | ChromaDB 0.5 |
| 嵌入模型 | BAAI/bge-m3 (HuggingFace) |
| 混合检索 | Dense Vector + BM25 + EnsembleRetriever |
| LLM | Ollama (qwen3.5:4b) |
| Agent | LangChain Agent + Tool |
| API | FastAPI + SSE + CORS |
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
- api.py                         FastAPI + SSE 流式 API（含 CORS）
- main.py                        CLI 交互入口
- evaluate.py                    RAGAS 评估脚本
- config.py                      配置数据类（环境变量覆盖）
- models.py                      LLM/Embedding 构建（支持远程 Ollama）
- rag-chat-frontend/             React + Vite 前端
- src/
  - components/
    - ChatWindow.jsx           聊天窗口（异步流式 + 思考计时）
    - DocumentManager.jsx      文档管理（单文件/文件夹上传 + 进度）
    - HistoryPanel.jsx         对话历史面板
  - App.jsx                    主应用（会话管理 + 侧边栏）
  - App.css                    全局样式
  - api.js                     后端 API 封装
- vite.config.js               Vite 构建配置
- requirements.txt               Python 依赖
- .gitignore

## 快速开始

### 前置依赖

- Python 3.11+
- Node.js 18+
- Redis（用于对话记忆，可选但推荐）
- Ollama（用于本地 LLM 推理）

### 安装与运行

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 启动 Redis（可选，用于对话记忆持久化）
# Windows: 下载并运行 redis-server.exe
# Linux/macOS: redis-server

# 3. 启动 Ollama 并拉取模型
ollama pull qwen3.5:4b

# 4. （可选）创建 .env 文件覆盖默认配置
cat > .env << 'EOF'
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_DEVICE=cuda:0
REDIS_URL=redis://localhost:6379/0
EOF

# 5. 启动 API 服务
python api.py

# 6. 启动前端（新开终端）
cd rag-chat-frontend
npm install
npm run dev

# 7. 浏览器打开 http://localhost:5173
```

### CLI 模式

```bash
python main.py your_document.pdf
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_MODEL` | `qwen3.5:4b` | Ollama 模型名称 |
| `OLLAMA_BASE_URL` | (默认本地) | Ollama 服务地址，支持远程 |
| `OLLAMA_TEMPERATURE` | `0.2` | 生成温度 |
| `OLLAMA_NUM_PREDICT` | `4096` | 最大生成 token 数 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型名称 |
| `EMBEDDING_DEVICE` | `cuda:0` | 嵌入模型设备（`cuda:0` / `cpu`） |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接地址 |
| `RAG_UPLOAD_DIR` | `./uploads` | 文档上传目录 |
| `RAG_INITIAL_PATH` | `test.pdf` | 启动时自动加载的初始文档路径 |

## 云部署

参见上方"环境变量"表格，关键步骤：

1. 云服务器安装 Python / Node.js / Redis / Ollama
2. 设置 `OLLAMA_BASE_URL` 指向 Ollama 服务
3. 设置 `EMBEDDING_DEVICE=cuda:0` 使用 GPU
4. 可选：用 `gunicorn` + `uvicorn` 运行后端，Nginx 托管前端静态文件
5. `VITE_API_BASE` 环境变量指定后端地址后进行 `npm run build`

## 评估

```bash
# 配置 DEEPSEEK_API_KEY 环境变量后运行
python evaluate.py
```
