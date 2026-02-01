import os 
import google.generativeai as genai
from yarl import Query
from pdfextractor import text_extractor
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS


st.markdown("""
<style>
.main {
    background: radial-gradient(circle at top, #0a122a, #05060f);
    color: #e8eaf0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070b1f, #05060f);
    border-right: 1px solid #1f2a44;
}

h1, h2, h3 {
    color: #e8eaf0;
}

.title-glow {
    text-shadow: 0 0 14px rgba(124,124,255,0.7);
}

textarea {
    background: #070b1f !important;
    color: #e8eaf0 !important;
    border-radius: 12px !important;
    border: 1px solid #7c7cff !important;
}

button {
    border-radius: 12px !important;
    background: linear-gradient(90deg, #7c7cff, #00e5ff) !important;
    color: #05060f !important;
    font-weight: 600 !important;
}

.chat-user {
    background: #070b1f;
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 12px;
    margin: 10px 0;
}

.chat-bot {
    background: #070b1f;
    border: 1px solid #00e5ff;
    border-radius: 12px;
    padding: 12px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 class="title-glow">
<span style="color:#7c7cff">ChatBot</span>
<span style="color:#00e5ff">AI</span>
</h1>
<p style="color:#9ca3af;margin-top:-10px">
AI-Assisted Document Chat using RAG
</p>
""", unsafe_allow_html=True)

st.info("""
**How to use:**
1. Upload a PDF from the sidebar  
2. Ask a question about the document  
3. Get instant AI-powered answers
""")


gemini_key = os.getenv('Google_API_Key2')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

embedding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')


st.sidebar.markdown("## 📄 Document Upload")
st.sidebar.caption("Only PDF files are supported")

pdf_file = st.sidebar.file_uploader('Upload here', type=['pdf'])

if not pdf_file:
    st.warning("Please upload a PDF from the sidebar to begin.")
    st.stop()

st.sidebar.success("Document uploaded successfully")
st.sidebar.progress(100)
st.sidebar.caption("You can now start chatting")


file_text = text_extractor(pdf_file)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(file_text)

vector_store = FAISS.from_texts(chunks, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={'k':3})


def gen_content(query):
    retrieved_docs = retriever.invoke(query)
    context = '\n'.join([d.page_content for d in retrieved_docs])

    augmented_prompt = f'''
    <Role> Your are an Helpful assistant using RAG.
    <Goal> Answer the question asked by the user. Here is the question: {query}
    <Context> Here are the documents retrived from the vector database to support the answer: {context}
    '''

    response = model.generate_content(augmented_prompt)
    return response.text


if 'history' not in st.session_state:
    st.session_state.history = []


st.subheader("💬 Chat")

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-user'>🧑 {msg['text']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bot'>🤖 {msg['text']}</div>",
            unsafe_allow_html=True
        )


with st.form('chatbot_form', clear_on_submit=True):
    user_query = st.text_area(
        "Ask a question about your document:",
        placeholder="e.g. Summarize chapter 2 in simple terms..."
    )
    send = st.form_submit_button('Send 🚀')


if user_query and send:
    st.session_state.history.append({'role':'user','text':user_query})

    with st.spinner("Thinking..."):
        answer = gen_content(user_query)

    st.session_state.history.append({'role':'ChatBot','text': answer})
    st.rerun()
