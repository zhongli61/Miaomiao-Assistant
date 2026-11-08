import streamlit as st
import streamlit.components.v1 as components
import itertools
from openai import OpenAI, AuthenticationError, RateLimitError
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*' # 或者明确的域名，如 'dashscope.aliyuncs.com'

# ================= 1. 配置区（智能 Key 池） =================

try:
    # 云端部署时从 Secrets 读取（注意：阿里云的 Key 变成了复数 API_KEYS）
    API_KEYS_STR = st.secrets["DASHSCOPE_API_KEYS"]
    API_KEYS = [k.strip() for k in API_KEYS_STR.split(",") if k.strip()]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
except Exception:
    # 本地测试用（在阿里云和 Qdrant 后台重置后的新 Key 填在这里）
    #第二个是花卷的，第一个是小狗撞地球的
    API_KEYS = [
        #"填入apikey",
        "填入apikey"

    ]
    QDRANT_URL = "填入你的Qdrant地址"
    QDRANT_API_KEY = "填入你的Qdrant密钥"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#CHAT_MODEL = "qwen3.7-plus"
#3.7plus虽然严谨但是太慢了，换一个快的演示
CHAT_MODEL = "qwen3.8-flash"
EMBEDDING_MODEL = "text-embedding-v2"
COLLECTION_NAME = "textbook_clean"  # 改成 clean，避开之前带广告的旧库

# ================= 2. 智能 Key 路由引擎 =================
if "active_keys" not in st.session_state:
    st.session_state.active_keys = API_KEYS.copy()


def safe_openai_call(func, *args, **kwargs):
    """自动切换 Key 的神器：遇到 401 或 429 自动拉黑并重试"""
    if not st.session_state.active_keys:
        st.error("🚨 所有的 API Key 额度均已耗尽或被限流，请联系管理员！")
        return None

    max_retries = len(st.session_state.active_keys)
    for _ in range(max_retries):
        current_key = st.session_state.active_keys[0]
        client = OpenAI(base_url=BASE_URL, api_key=current_key)

        try:
            return func(client, *args, **kwargs)
        except (AuthenticationError, RateLimitError) as e:
            # 拦截 401(失效) 或 429(限流/欠费) 错误
            st.warning(f"⚠️ Key 尾号 {current_key[-4:]} 额度耗尽，自动切换下一个...")
            st.session_state.active_keys.pop(0)
            if not st.session_state.active_keys:
                st.error("🚨 所有 Key 均已阵亡！")
                return None
        except Exception as e:
            # 其他错误（如网络超时）直接抛出
            raise e


def get_embedding(text):
    res = safe_openai_call(lambda c: c.embeddings.create(input=[text], model=EMBEDDING_MODEL))
    if res: return res.data[0].embedding
    return [0.0] * 1536


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
        field_schema=models.PayloadSchemaType.KEYWORD, # 指定为 keyword 类型
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
    uploaded_files = st.file_uploader("上传 PDF 文件", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("确认存入云端数据库"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在解析并上传《{uploaded_file.name}》..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    loader = PyMuPDFLoader(temp_path)
                    docs = loader.load()

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

                        # 👇 用 safe_openai_call 替换原来的 client
                        summary_res = safe_openai_call(
                            lambda c: c.chat.completions.create(
                                model=CHAT_MODEL,
                                messages=[{"role": "user",
                                           "content": f"请根据以下开头片段，用两三句话总结这本书主要讲什么（用中文）：\n{preview_text}"}]
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
    st.header("书籍导读")
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


    # 【修改点4：智能过滤盲盒（过滤目录和废话）】
    st.markdown("---")
    st.header("🎲 知识盲盒")
    st.write("不知道问什么？随机抽个原文看看：")

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
                        "我提供了5段不同的教材片段。请为每一段分别提出一个简单、接地气的中文提问建议。\n"
                        "⚠️【严格要求】：\n"
                        "1. 绝对不要使用“这段例子”、“这个定理”、“上述”、“该”等任何缺乏上下文的指示代词。\n"
                        "2. 如果片段里提到“例子”、“绿色”、“图1”等具体场景，你必须提取出它的核心学术概念来提问。\n"
                        "3. 提出来的问题必须是一个独立、完整的句子，让没看过书的人也能一眼看懂要问什么。\n"
                        "4. 严格按照以下格式输出5行内容，不要有任何废话：\n"
                        "1. [第一个提问]\n2. [第二个提问]\n3. [第三个提问]\n4. [第四个提问]\n5. [第五个提问]\n\n"
                        f"文本如下：\n{combined_text}"
                    )

                    check_res = safe_openai_call(lambda c: c.chat.completions.create(model=CHAT_MODEL,
                        messages=[{
                            "role": "user",
                            "content": prompt_instruction
                        }]
                    ))
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

                # 检索前 8 条
                response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vec,
                    limit=8,
                    query_filter=Filter(must_not=[FieldCondition(key="type", match=MatchValue(value="guide"))]) # 过滤掉导读片段，防止AI把导读当成知识点检索出来
                )
                search_result = response.points

                context = "\n".join([hit.payload["text"] for hit in search_result])

                prompt_text = f"""你是一个专业的学术助手。请根据以下资料片段回答用户的问题。
                如果遇到类似“A ⊠ B”的乱码符号，请结合上下文（比如集合的数字）推测它是数学交集（∩）或并集（∪）。
                如果资料没有提及，请如实回答。
资料：
{context}

用户问题：{prompt}"""

                response = safe_openai_call(lambda c: c.chat.completions.create(model=CHAT_MODEL, messages=[
                    {"role": "user", "content": prompt_text}],stream=True)) #开启流式会更快显示
                if response:
                    # 使用 st.write_stream 直接把流水接住并显示在屏幕上
                    answer = st.write_stream(response)
                    # 存入历史记录
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("API 调用失败，请检查额度或网络。")

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
