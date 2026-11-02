import streamlit as st
from openai import OpenAI
import chromadb

# ================= 1. 配置区 =================
# 这里先留空，我们在云端会通过 Secrets 填入
# 如果你本地要测试，可以填你的 Key
import os

# 获取 API Key (支持本地写死，也支持云端读环境变量)
try:
    # 尝试从云端 Secrets 读取
    API_KEY = st.secrets["DASHSCOPE_API_KEY"]
except Exception:
    # 本地运行时，如果没配置 Secrets，就用你写死的默认 Key
    API_KEY = "填入你的API_KEY"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen3.7-plus"
EMBEDDING_MODEL = "text-embedding-v2"

# 初始化 OpenAI 客户端
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# 获取向量的函数
def get_embedding(text):
    res = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return res.data[0].embedding


# ================= 2. 加载数据库 =================
@st.cache_resource
def load_db():
    # 连接到本地的 chromaDB 文件夹
    chroma_client = chromadb.PersistentClient(path="chromaDB")
    collection = chroma_client.get_collection(name="demo001")
    return collection


try:
    collection = load_db()
except Exception as e:
    st.error(f"读取数据库失败，请确保 chromaDB 文件夹存在: {e}")

# ================= 3. 网页界面 =================
st.set_page_config(page_title="健康档案AI助手", page_icon="🐱")
st.title("🐱我是助手小猫")
st.caption("基于 RAG 技术的私有知识库问答系统")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ================= 4. 聊天逻辑 =================
if prompt := st.chat_input("请输入你想问的问题，例如：张三九的血糖是多少？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在检索档案并思考中..."):
            try:
                # 1. 把问题变成向量
                query_vec = get_embedding(prompt)

                # 2. 去 Chroma 里找最相似的 3 段文本
                results = collection.query(query_embeddings=[query_vec], n_results=3)
                context = "\n".join(results['documents'][0])

                # 3. 组装 Prompt
                prompt_text = f"""你是一个专业的健康助手，请根据以下的健康档案片段回答用户的问题。
如果档案中没有提到，请说“档案中未提及”。
健康档案片段：
{context}

用户问题：{prompt}
"""

                # 4. 调用大模型生成回答
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt_text}],
                    stream=False
                )
                answer = response.choices[0].message.content

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"出错了: {e}")