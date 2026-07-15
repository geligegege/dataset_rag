<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=34&duration=2800&pause=900&color=27AE60&center=true&vCenter=true&width=760&lines=Dataset+RAG;Build+Your+Intelligent+Knowledge+Base" alt="Dataset RAG animated title" />

### 把一堆不会说话的资料，变成一个随时能问的知识库

产品手册太厚、技术文档太散、关键词永远搜不准？把文件交给它，剩下的事情让知识库自己回答。

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangGraph-Agent-1C3C3C" alt="LangGraph" />
  <img src="https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Milvus-Hybrid_Search-00A1EA" alt="Milvus" />
  <img src="https://img.shields.io/badge/BGE--M3-Embedding-F5A623" alt="BGE-M3" />
  <img src="https://img.shields.io/badge/Status-Developing-orange" alt="Status" />
</p>

<p>
  <a href="#-核心能力">核心能力</a> ·
  <a href="#-工作流程">工作流程</a> ·
  <a href="#-快速开始">快速开始</a> ·
  <a href="#-查询示例">查询示例</a>
</p>

</div>

---

> **这个项目想解决什么？**  
> 不是让人翻遍几十份 PDF 找一句话，而是让资料主动把答案送到面前。你负责提问，它负责读文档、找重点、排除干扰，再组织成一份能看懂的回答。

## ✨ 核心能力

可以把它理解成一位“读过公司全部资料，而且不会嫌你问题多”的知识库助手：

- LangGraph 分别编排知识导入与查询工作流。
- MinerU 解析 PDF，支持 Markdown 与图片处理。
- BGE-M3 生成稠密/稀疏向量，Milvus 执行混合检索。
- 普通向量、HyDE 与 MCP 联网搜索多路召回。
- RRF 融合结果，BGE Reranker 二次精排。
- MinIO 保存文档图片，MongoDB 保存会话历史。
- FastAPI 提供导入、查询、任务状态和 SSE 流式接口。
- 可选接入 Neo4j 扩展知识图谱检索。

## 🔄 工作流程

```text
导入：PDF/MD → 解析 → 图片处理 → 切片 → 实体识别 → BGE-M3 → Milvus
查询：问题 → 实体确认 → 多路召回 → RRF → Rerank → 生成答案 → SSE
```

## 🧰 技术栈

| 模块 | 技术 |
| --- | --- |
| 工作流 | LangGraph、LangChain |
| API | FastAPI、Uvicorn、SSE |
| 文档解析 | MinerU / Magic PDF |
| 检索 | BGE-M3、Milvus、BGE Reranker |
| 存储 | MinIO、MongoDB、Neo4j（可选） |
| 大模型 | OpenAI 兼容接口 |

## 🚀 快速开始

要求 Python 3.12+、uv、CUDA 12.8 兼容环境，以及 Milvus、MongoDB、MinIO。

```bash
uv sync
cp .env.example .env
```

编辑 `.env`，配置大模型、Milvus、模型路径、MongoDB 与 MinIO。随后可下载本地模型：

```bash
uv run python -m app.tool.download_bgem3
uv run python -m app.tool.download_reranker
```

启动查询与导入服务：

```bash
chmod +x start_dataset_rag.sh
./start_dataset_rag.sh start
```

| 服务 | 默认端口 | 页面/接口 |
| --- | ---: | --- |
| 查询服务 | 8080 | `/chat.html`、`/query`、`/stream/{session_id}` |
| 导入服务 | 8081 | `/import`、`/upload`、`/status/{task_id}` |

管理命令：`start`、`stop`、`restart`、`status`、`logs`。端口可通过 `QUERY_PORT` 和 `IMPORT_PORT` 覆盖。

## 💬 查询示例

```json
{
  "session_id": "demo-session",
  "query": "HAK180 产品有哪些安全注意事项？",
  "is_stream": false
}
```

请求 `POST /query`。流式模式提交 `is_stream: true` 后连接 `GET /stream/{session_id}`。

## 🔐 安全建议

- 不要提交 `.env`、模型权重、日志、上传文件及解析产物。
- 生产环境应限制 CORS，并为上传、查询和历史记录接口增加认证。
- 对上传文件增加格式、大小和恶意内容检查。
- 外部搜索与模型回答应保留来源和审计记录。

## 📌 项目状态

项目处于开发阶段，适合企业知识库原型验证与二次开发。
