import streamlit as st
import streamlit.components.v1 as components
import itertools
from openai import OpenAI, AuthenticationError, RateLimitError
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import os

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'  # 或者明确的域名，如 'dashscope.aliyuncs.com'

# ================= 1. 配置区（智能 模型+Key 双引擎） =================
try:
    # 云端部署时从 Secrets 读取
    API_KEYS_STR = st.secrets["DASHSCOPE_API_KEYS"]
    API_KEYS = [k.strip() for k in API_KEYS_STR.split(",") if k.strip()]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

    # 如果想在云端配置 DeepSeek，防止模型崩溃对实际工程的影响，可以加这个
    DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
except Exception:
    # 本地测试用（在这里填入key）

    API_KEYS = [
        "api_key1", # 小狗撞地球的
        "api_key2" # 花卷的
    ]
    QDRANT_URL = "https://your-qdrant-cluster-url:6333"
    QDRANT_API_KEY = "your-qdrant-api-key"
    DEEPSEEK_API_KEY = "your-deepseek-api-key"  # 如果有 DeepSeek 的 key 填这里

EMBEDDING_MODEL = "text-embedding-v2"
COLLECTION_NAME = "textbook_clean"

# 核心升级：定义模型优先级链（从上到下尝试）
MODEL_CHAIN = [
    {
        "name": "千问极速版",
        "model_id": "qwen3.8-flash",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "keys": API_KEYS
    }
]

# 如果有 DeepSeek 的 Key，加入备选链
if DEEPSEEK_API_KEY:
    MODEL_CHAIN.append({
        "name": "DeepSeek备用版",
        "model_id": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "keys": [DEEPSEEK_API_KEY]
    })
# 如果包里没有key，就把我的key加进去
if "active_keys" not in st.session_state:
    st.session_state.active_keys = API_KEYS.copy()

# ================= 2. 智能 Key 路由引擎 =================
# 注意：这里不再全局保存 active_keys，因为换模型时 Key 也要跟着换

def safe_openai_call(func, *args, **kwargs):
    last_error = None
    for model_config in MODEL_CHAIN:
        current_keys = model_config["keys"]
        if not current_keys:
            continue
        for current_key in current_keys:
            client = OpenAI(base_url=model_config["base_url"], api_key=current_key)
            # 强制覆盖模型名，让对话模型可以降级
            kwargs['model'] = model_config["model_id"]
            try:
                return func(client, *args, **kwargs)
            except (AuthenticationError, RateLimitError) as e:
                st.warning(f"⚠️ {model_config['name']} 的 Key 尾号 {current_key[-4:]} 额度耗尽，尝试切换...")
                last_error = e
                continue
            except Exception as e:
                st.warning(f"⚠️ {model_config['name']} 调用失败，原因: {str(e)}")
                last_error = e
                break

    st.error(f"🚨 所有的 API Key 及备用模型均已阵亡。最后一次报错信息为：{str(last_error)}")
    return None


def get_embedding(text):
    last_err = None
    # 只轮询 Key，强制使用 text-embedding-v2
    for key in st.session_state.active_keys:
        try:
            client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=key)
            res = client.embeddings.create(input=[text], model="text-embedding-v2")
            return res.data[0].embedding
        except Exception as e:
            last_err = e
            continue
    # 如果全部 Key 都失败，直接抛出异常，中断上传，绝不让脏数据入库
    raise Exception(f"向量计算失败，所有 Key 均不可用: {str(last_err)}")


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120  # 将超时时间延长到120秒,防止文件上传的时候因为超时而失败
    )


qdrant = get_qdrant_client()

# 初始化集合（如果没有就自动建一个 1536 维的）
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
# ================= 新增：为 "type" 字段创建 keyword 索引 =================
try:
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="type",
        field_schema=models.PayloadSchemaType.KEYWORD,  # 指定为 keyword 类型
        wait=True  # 等待索引创建完成
    )
    print(f"已为 '{COLLECTION_NAME}' 集合的 'type' 字段创建索引")
except Exception as e:
    # 如果索引已存在，Qdrant 不会报错，这里只是为了捕获其他意外情况
    print(f"ℹ索引可能已存在或创建时遇到提示: {e}")

# ================= 2. 侧边栏：上传文件功能 =================
with st.sidebar:
    st.header("📂 知识库管理")

    # 【修改点1：允许上传多个文件（不要求一次只传一本）】
    uploaded_files = st.file_uploader("上传文档 (支持 PDF/TXT/DOCX)", type=["pdf", "txt", "docx", "caj"],
                                      accept_multiple_files=True)

    if uploaded_files and st.button("确认存入云端数据库"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在解析并上传《{uploaded_file.name}》..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    #支持的几种格式，不支持caj但识别
                    import os
                    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader

                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

                    if file_ext == ".pdf":
                        loader = PyMuPDFLoader(temp_path)
                        docs = loader.load()
                    elif file_ext == ".txt":
                        loader = TextLoader(temp_path, encoding="utf-8")
                        docs = loader.load()
                    elif file_ext == ".docx":
                        loader = Docx2txtLoader(temp_path)
                        docs = loader.load()
                    elif file_ext == ".caj":
                        st.error(f"⚠️ 《{uploaded_file.name}》是 CAJ 格式，请先转成 PDF 再上传哦！")
                        os.remove(temp_path)
                        continue
                    else:
                        st.error("不支持该格式！")
                        os.remove(temp_path)
                        continue

                    # ================= 新增：清洗每页的广告/页眉页脚 =================
                    # 只通过精准关键词过滤，绝对不按长度误杀正文
                    ad_keywords = ["关注公众号", "扫码", "微信", "QQ群", "获取更多", "本文由", "版权所有", "科研帝论坛",
                                   "更多数据分析英文电子书", "公众号", "www.", " 加V", "加v"]

                    for doc in docs:
                        lines = doc.page_content.split('\n')
                        # 如果这一行包含任何一个广告词，就把它干掉
                        clean_lines = [line for line in lines if not any(kw in line for kw in ad_keywords)]

                        # 把过滤后的行拼回去，顺便修复一下乱码符号
                        cleaned_text = '\n'.join(clean_lines)
                        cleaned_text = cleaned_text.replace("⊠", "∩")

                        doc.page_content = cleaned_text

                    # ==================================================================

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=50,
                        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
                    )
                    chunks = text_splitter.split_documents(docs)

                    # ====== 加回进度条 ======
                    total_chunks = len(chunks)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    points = []
                    for i, chunk in enumerate(chunks):
                        vec = get_embedding(chunk.page_content)
                        points.append(PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload={
                                "text": chunk.page_content,
                                "page": chunk.metadata.get("page", 0) + 1,
                                "source": uploaded_file.name
                            }
                        ))
                        # 更新进度条
                        if (i + 1) % 5 == 0 or (i + 1) == total_chunks:
                            status_text.text(f"正在计算向量：{i + 1} / {total_chunks} 段...")
                            progress_bar.progress((i + 1) / total_chunks)

                    # 分批写入 Qdrant
                    BATCH_SIZE = 100
                    for i in range(0, len(points), BATCH_SIZE):
                        batch = points[i:i + BATCH_SIZE]
                        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True, timeout=120)

                    os.remove(temp_path)
                    status_text.text("向量已入库！")
                    st.success(f"✅ 《{uploaded_file.name}》已存入云端！")

                    # ====== 生成导读 ======
                    with st.spinner("正在生成导读..."):
                        preview_text = "\n".join([c.page_content for c in chunks[:3]])

                        summary_res = safe_openai_call(
                            lambda c, **kw: c.chat.completions.create(
                                messages=[{"role": "user",
                                           "content": f"请根据以下开头片段，用两三句话总结这本书主要讲什么（用中文）：\n{preview_text}"}],
                                **kw
                            )
                        )

                        if summary_res:
                            guide_text = summary_res.choices[0].message.content
                        else:
                            guide_text = "导读生成失败，请稍后重试。"

                        # 下面是存入数据库的代码，保持不变
                        guide_point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=get_embedding("书籍导读占位文本"),
                            payload={
                                "text": guide_text,
                                "source": uploaded_file.name,
                                "type": "guide"
                            }
                        )
                        qdrant.upsert(collection_name=COLLECTION_NAME, points=[guide_point])

                        if "book_guides" not in st.session_state:
                            st.session_state["book_guides"] = {}
                        st.session_state["book_guides"][uploaded_file.name] = guide_text

                except Exception as e:
                    st.error(f"上传失败: {e}")

    st.markdown("---")

    # 【修改点3：按书籍分类展示导读】
    st.header("📚 书籍导读")
    # 每次加载网页，去 Qdrant 里查出所有 type="guide" 的导读
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        guides, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="guide"))]),
            limit=20,
            with_payload=True
        )

        if guides:
            # 用字典去重（防止同一本书重复生成导读）
            unique_guides = {g.payload["source"]: g.payload["text"] for g in guides}
            for book_name, guide in unique_guides.items():
                with st.expander(f"📖 {book_name}", expanded=False):
                    st.info(guide)
        else:
            st.caption("暂无书籍导读。")
    except Exception as e:
        st.caption("导读加载中...")

    # 新增：一键修复导读按钮
    if st.button("🔧 一键检查并补全导读", use_container_width=True):
        with st.spinner("正在翻阅云端数据库排查缺失的导读..."):
            try:
                # 1. 把库里所有的文本和导读都拿出来
                records, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=2000, with_payload=True)

                all_sources = {}
                for r in records:
                    src = r.payload.get("source")
                    typ = r.payload.get("type")
                    text = r.payload.get("text", "")
                    if not src: continue

                    if src not in all_sources:
                        all_sources[src] = {"guide_text": None, "texts": [], "bad_id": None}

                    if typ == "guide":
                        # 如果之前存的是失败信息，也算作没导读
                        if "导读生成失败" in text:
                            all_sources[src]["bad_id"] = r.id
                        else:
                            all_sources[src]["guide_text"] = text
                    else:
                        # 收集一些正文作为生成导读的材料
                        if len(all_sources[src]["texts"]) < 3:
                            all_sources[src]["texts"].append(text)

                # 2. 找出没有导读或者导读坏掉的书籍
                missing_sources = [src for src, info in all_sources.items() if not info["guide_text"]]

                if not missing_sources:
                    st.success("✅ 太棒了，所有文档都有完好的导读，无需修复！")
                else:
                    # 3. 逐个补全
                    for src in missing_sources:
                        # 如果存在坏的导读，先删掉
                        if all_sources[src]["bad_id"]:
                            qdrant.delete(collection_name=COLLECTION_NAME,
                                          points_selector=[all_sources[src]["bad_id"]])

                        preview_text = "\n".join(all_sources[src]["texts"])
                        if not preview_text:
                            continue

                        summary_res = safe_openai_call(
                            lambda c,**kw: c.chat.completions.create(
                                messages=[{"role": "user",
                                           "content": f"请根据以下开头片段，用两三句话总结这本书主要讲什么（用中文）：\n{preview_text}"}],
                                **kw
                            )
                        )
                        guide_text = summary_res.choices[0].message.content if summary_res else "导读生成失败，请重试。"

                        guide_point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=get_embedding("书籍导读占位文本"),
                            payload={"text": guide_text, "source": src, "type": "guide"}
                        )
                        qdrant.upsert(collection_name=COLLECTION_NAME, points=[guide_point])

                    st.success(f"🎉 成功修复了 {len(missing_sources)} 本书的导读！")
                    st.rerun()  # 刷新网页，立刻显示出来
            except Exception as e:
                st.error(f"修复失败：{e}")

    #限制搜索范围，支持书籍多选
    st.markdown("---")
    st.header("🎯 检索范围")

    # 去数据库里把所有不重复的书名（source）抓出来
    try:
        book_records, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=1000, with_payload=["source"])
        # 用集合去重，并过滤掉空值
        book_names = list(set([r.payload.get("source") for r in book_records if r.payload.get("source")]))
        book_names.sort()
    except Exception:
        book_names = []

    # 生成下拉菜单，默认是全库检索
    selected_books = st.multiselect(
        "选择要查询的书籍（可多选，不选默认全库）：",
        options=book_names,
        default=None  # 默认什么都不选，代表全选
    )




    # 【修改点4：智能过滤盲盒（过滤目录和废话）】
    st.markdown("---")
    st.header("🎲 知识盲盒")
    st.write("不知道问什么？随机抽个原文看看：")

    # 先在按钮前面放一个“看不见的占位符”，防止开盲盒的时候出现残影
    blind_box_placeholder = st.empty()

    if st.button("开盲盒（抽5个）", use_container_width=True):
        with st.spinner("正在翻阅数据库并生成5个建议..."):
            try:
                records, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=30,
                    with_payload=True,
                    scroll_filter=Filter(must_not=[FieldCondition(key="type", match=MatchValue(value="guide"))])
                )

                if len(records) >= 5:
                    import random

                    random_records = random.sample(records, 5)

                    combined_text = ""
                    for i, r in enumerate(random_records):
                        combined_text += f"【片段{i + 1}】\n{r.payload.get('text', '')[:300]}\n\n"

                    # 核心修复点：使用规范的 prompt 拼接格式
                    prompt_instruction = (
                        "你是一个经验丰富的统计学助教，你的任务是根据提供的5段教材片段，分别提出一个极其专业、独立的中文提问建议。\n\n"
                        "⚠️【必须遵守的铁律】：\n"
                        "1. **绝对禁止**使用任何代词！比如：不要写“这段例子”、“这个定理”、“上述”、“该方法”、“绿色的一面”、“这里”、“文中”。\n"
                        "2. 你的问题必须是一个**完整的独立问题**，让完全没有看过书的人也能一眼看懂。\n"
                        "3. 如果原文片段只有具体的数字或案例（比如：抛硬币、抽球），请结合上下文，把它们**泛化**成对应的学术概念来提问（比如：条件概率、古典概型）。\n\n"
                        "【正反例示范】：\n"
                        "反例（错误）：为什么看到一面是绿色，另一面也是绿色的概率不是1/2而是2/3？\n"
                        "正例（正确）：在“蒙提霍尔问题（三门问题）”的条件概率模型中，为什么换门后的获胜概率是2/3？\n\n"
                        "反例（错误）：在这个例子中，X的期望是怎么算的？\n"
                        "正例（正确）：对于离散型随机变量，其数学期望的计算公式是什么？\n\n"
                        "请严格按照以下格式输出5行内容，不要有任何废话（不要写“好的”、“以下是”）：\n"
                        "1. [独立完整的问题]\n"
                        "2. [独立完整的问题]\n"
                        "3. [独立完整的问题]\n"
                        "4. [独立完整的问题]\n"
                        "5. [独立完整的问题]\n\n"
                        f"教材片段如下：\n{combined_text}"
                    )

                    check_res = safe_openai_call(
                        lambda c, **kw: c.chat.completions.create(
                            messages=[{
                                "role": "user",
                                "content": prompt_instruction
                            }],
                            **kw
                        )
                    )
                    if check_res: reply = check_res.choices[0].message.content

                    suggestions = [line.strip() for line in reply.split('\n') if line.strip()]
                    import re

                    suggestions = [re.sub(r'^\d+[\.\、]\s*', '', s) for s in suggestions[:5]]

                    st.session_state["random_snippets"] = suggestions
                else:
                    st.session_state["random_snippets"] = []
            except Exception as e:
                st.error(f"出错了: {e}")

    # 极简版卡片：只保留推荐问题和复制按钮
    # 展示 5 个盲盒卡片
    if "random_snippets" in st.session_state and st.session_state["random_snippets"]:
        import json

        suggestions_list = st.session_state["random_snippets"]

        # 全新的 CSS 样式：上下布局，完美防挤压
        html_content = """
        <style>
            .suggestion-card {
                background-color: #ffffff; border: 1px solid #e0e0e0; 
                padding: 12px 15px; border-radius: 8px; margin-bottom: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-family: sans-serif;
            }
            .suggestion-text {
                font-size: 14px; color: #31333F; line-height: 1.5; margin-bottom: 10px;
                word-wrap: break-word; word-break: break-all; /* 强制长英文字符或公式换行 */
            }
            .btn-container {
                text-align: right; /* 按钮靠右对齐 */
            }
            .copy-btn {
                background-color: #FF4B4B; color: white; border: none; 
                padding: 6px 12px; border-radius: 5px; cursor: pointer;
                font-weight: bold; font-size: 12px;
                transition: 0.2s;
            }
            .copy-btn:hover { background-color: #ff3333; }
        </style>
        """

        for i, text in enumerate(suggestions_list):
            safe_text = json.dumps(text)
            html_content += f"""
            <div class="suggestion-card">
                <div class="suggestion-text">{i + 1}. {text}</div>
                <div class="btn-container">
                    <button class="copy-btn" onclick='copyToClipboard({safe_text}, this)'>📋 复制</button>
                </div>
            </div>
            """

        # JS 代码保持不变
        html_content += """
        <script>
        function copyToClipboard(text, btnElement) {
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(text).then(() => { showSuccess(btnElement); }).catch(err => { fallbackCopy(text, btnElement); });
            } else { fallbackCopy(text, btnElement); }
        }
        function fallbackCopy(text, btnElement) {
            var textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed"; textArea.style.top = "-9999px"; textArea.style.left = "-9999px";
            document.body.appendChild(textArea); textArea.focus(); textArea.select();
            try {
                if (document.execCommand('copy')) { showSuccess(btnElement); } else { alert('复制失败，请手动选中复制。'); }
            } catch (err) { alert('浏览器限制，请手动复制。'); }
            document.body.removeChild(textArea);
        }
        function showSuccess(btnElement) {
            let originalText = btnElement.innerHTML;
            btnElement.innerHTML = "✅ 成功";
            btnElement.style.backgroundColor = "#28a745";
            setTimeout(() => { btnElement.innerHTML = originalText; btnElement.style.backgroundColor = "#FF4B4B"; }, 1500);
        }
        </script>
        """

        # 🎯 重新设计的动态高度计算（更保守，防截断）
        total_height = 20  # 基础高度
        for text in suggestions_list:
            # 按每行最多 15 个字符估算（留足余量）
            lines = (len(text) // 15) + 1
            # 每行占 22px 高度
            text_height = lines * 22

            # 卡片总高 = 文本高度 + 按钮高度(30) + Padding上下(24) + Margin(12)
            card_height = text_height + 30 + 24
            total_height += card_height + 12

        # 额外加上 30px 的安全冗余，确保最后一个卡片绝不会被切掉
        total_height += 30

        components.html(html_content, height=total_height)
# ================= 3. 网页主界面 =================
st.set_page_config(page_title="教材 AI 助手", page_icon="🐱")
st.title("🐱 专属专业课 AI 助手")
st.caption("云端持久化 | 中英双语 | 溯源展示")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar_icon = "🦉" if message["role"] == "user" else "🍬"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# 接收输入部分
if prompt := st.chat_input("请输入你想问的问题..."):
    with st.chat_message("user", avatar="🦉"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🍬"):
        with st.spinner("正在云端检索并思考中..."):
            try:
                query_vec = get_embedding(prompt)

                # 1. 准备过滤条件
                filter_must_not = [FieldCondition(key="type", match=MatchValue(value="guide"))]

                # 注意这里
                filter_must = []
                if selected_books:  # 如果用户选了具体的书（多选列表不为空）
                    # 使用 MatchAny，只要来源是这几本书里的任意一本，就命中
                    filter_must.append(FieldCondition(key="source", match=MatchAny(any=selected_books)))

                # 2. 开始检索
                response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vec,
                    limit=8,
                    query_filter=Filter(
                        must_not=filter_must_not,
                        must=filter_must if filter_must else None
                    )
                )
                search_result = response.points

                # ================= 开发者调试面板 =================
                # 如果想在线上彻底隐藏，就把下面这行里的 True 改成 False
                if True:
                    with st.expander("🛠️ 开发者调试面板（点击查看检索明细）", expanded=False):
                        for i, hit in enumerate(search_result):
                            st.markdown(
                                f"**匹配度得分 (Score)**: `{hit.score:.4f}` | **来源**: 《{hit.payload.get('source')}》 第 {hit.payload.get('page')} 页")
                            # 只显示前 150 个字符，防止太长
                            st.text_area(f"片段 {i + 1} 内容：", hit.payload["text"], height=100, disabled=True,
                                         key=f"debug_{i}")
                # ====================================================

                context = "\n".join([hit.payload["text"] for hit in search_result])

                prompt_text = f"""你是一个专业的学术助手。请根据以下资料片段回答用户的问题。

                ⚠️【数学公式排版要求】：如果回答中包含数学公式，**绝对不要使用 \( \) 或 \[ \] 作为定界符！**
                请统一使用美元符号：行内公式用 `$公式$`，独立公式用 `$$公式$$`。

                如果资料没有提及，请如实回答。
                资料：
                {context}
                用户问题：{prompt}"""

                # 1. 开启流式--不用streamlit的封装，防止网络缓冲/key切换的干扰
                response = safe_openai_call(
                    lambda c, **kw: c.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_text}],
                        stream=True,
                        **kw
                    )
                )

                if response:
                    # 2. 手动创建一个空白占位符
                    placeholder = st.empty()
                    answer = ""

                    try:
                        # 3. 手动遍历流式对象的每一小块
                        for chunk in response:
                            # 取出这一块的文字
                            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                                answer += chunk.choices[0].delta.content
                                # 每次多出一点字，就立刻刷新页面（带光标效果）
                                placeholder.markdown(answer + "▌")

                        # 结束后，去掉光标，显示完整答案
                        placeholder.markdown(answer)

                        # 存入历史
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as stream_err:
                        st.error(f"流式输出意外中断: {stream_err}")
                else:
                    st.error("API 调用失败，请稍后重试。")

                # ================= 溯源展示 =================
                st.markdown("---")
                st.markdown("🔍 **参考资料溯源：**")
                for i, hit in enumerate(search_result):
                    page_num = hit.payload.get("page", 1)
                    source_name = hit.payload.get("source", "未知文档")
                    st.info(f" 来自《{source_name}》 第 {page_num} 页")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"出错了: {e}")
