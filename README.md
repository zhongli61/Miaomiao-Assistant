# 专属专业课 AI 助手 (RAG-based Academic Assistant)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

一个基于 **RAG (检索增强生成)** 架构的云端学术知识库问答系统。旨在为大学生和研究人员提供一个专属的“私人助教”。用户上传教材/文献后，系统能精准检索并提供带有原文页码溯源的中文解答。

 **演示视频**: [B站/YouTube链接](视频链接)

## 核心功能与工程亮点 (Key Features)

### 智能检索与交互体验
- **全格式文档接入**：支持 `PDF`, `TXT`, `DOCX` 等多种格式上传，自动过滤冗余页眉页脚与广告词，智能处理中英文混排切分。
- **元数据多选过滤 (Metadata Filtering)**：支持多选指定教材/书籍进行定向检索，彻底解决多领域知识混合时的“串台”问题。
- **开发者调试与精准溯源**：回答下方附带“参考资料溯源”（指明第几页），并内置 Debug 面板，实时展示底层文本块的相似度得分 (Similarity Score)。
- **流式输出 (Streaming)**：基于大模型流式接口实现打字机效果，大幅降低用户等待焦虑。

### 高可用性架构 (High Availability)
- **多 Key 负载均衡池**：针对个人开发者 API 额度有限的问题，设计了带故障转移 (Failover) 机制的 Key 轮询池。某个 Key 触发限流时自动拉黑并无缝切换。
- **多模型降级链**：配置模型优先级链（如优先调用 Qwen-Flash，若服务异常则自动降级到 DeepSeek），保证系统全天候稳定可用。
- **云端持久化**：摒弃本地内存存储，采用 Qdrant Cloud 向量数据库，确保数据不丢失，且支持多人同时在线访问。

### 辅助创新功能
- **智能书籍导读**：文档上传后，自动提取开头核心段落，调用大模型生成中文导读并永久存入数据库。
- **知识盲盒 (Knowledge Blind Box)**：随机抓取库内知识点，利用提示词工程 (Prompt Engineering) 生成 5 个无上下文缺失的专业提问建议，并提供前端一键复制按钮，解决用户“不知道问什么”的痛点。

## 技术架构 (Tech Stack)

- **前端框架**：Streamlit (实现快速原型与流式 UI)
- **大模型层**：通义千问 (Qwen), DeepSeek API (兼容 OpenAI SDK)
- **向量模型**：Text-Embedding-v2 (阿里云 DashScope)
- **向量数据库**：Qdrant Cloud (支持过滤、持久化及高并发)
- **文档处理**：PyMuPDF, LangChain Text Splitters

## 本地运行指南 (Quick Start)

如果你想在本地运行本项目，请按照以下步骤操作：

1. **克隆仓库**
   ```bash
   git clone https://github.com/你的用户名/你的仓库名.git
   cd 你的仓库名

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
3. **配置环境变量**
   在根目录创建 .streamlit/secrets.toml 文件，填入你的密钥（请勿提交到 GitHub）：
   ```toml
   DASHSCOPE_API_KEYS = "你的阿里云key1,你的阿里云key2"
   QDRANT_URL = "你的Qdrant集群地址"
   QDRANT_API_KEY = "你的Qdrant密钥"
   DEEPSEEK_API_KEY = "你的DeepSeek密钥(可选)"
5. **启动应用**
   ```bash
   streamlit run app.py

## 一点踩坑与痛点记录

痛点：大模型直接读取长文档时容易超出 Token 限制且找不到重点。
解决：采用 RecursiveCharacterTextSplitter 进行递归切分，并在自定义 Prompt 中强制要求大模型使用 $公式$ 的 LaTeX 格式，解决数学公式在前端渲染乱码的问题。

痛点：多人使用免费 API Key 极易触发 429 限流。
解决：手写 safe_openai_call 闭包，捕获 RateLimitError 异常并实现 Key 级别的轮询与模型级别的降级。

痛点：PDF 解析时常带有牛皮癣广告，污染向量库。
解决：在加载器 load() 之后引入正则清洗层，精准剔除特定关键词，大幅节省 Embedding Token 消耗。

本项目作为计算机/数据科学课程期末大作业开发，旨在探索大模型在实际学术场景中的工程化落地。
