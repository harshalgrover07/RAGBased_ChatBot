import os 
import google.generativeai as genai
from yarl import Query
from pdfextractor import text_extractor
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS


# Imporved UI

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



# Lets configure the models
# Step 1 : Lets  configure the Model

# LLM MODEL
gemini_key = os.getenv('Google_API_Key2')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Configure the embedding model 

embedding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

# Lets Create the the main page

st.title(':orange[ChatBot:]:blue[AI Assisted chatbot using RAG]')
tips ='''
Follow The Steps to use the appliction:
* Upload your PDF Documnets in sidebr.
* Write a query and start a chat. 
'''
st.text(tips)

# Lets create the sidebar
st.sidebar.title(':green[Upload Your File:]')
st.sidebar.subheader('Upload PDF file only.')
pdf_file = st.sidebar.file_uploader('Upload here',type=['pdf'])
if pdf_file:
    st.sidebar.success('File uploaded successfully')

    file_text = text_extractor(pdf_file)
    
    # Step 1 chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Step 2 : Create vector store database(FAISS)

    vector_store = FAISS.from_texts(chunks,embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})
    
    def gen_content(query):
        #Step 3 : retrieval (R)
        retrieved_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrieved_docs])

        # step 4 : 

        augmented_prompt = f'''
        <Role> Your are an Helpful assistant using RAG.
        <Goal> Answer the question asked by the user. Here is the question: {query}
        <Context> Here are the documents retrived from the vector  database to support the answer which you have to generate {context}

        '''

        # Step 5 : Generate :
        response = model.generate_content(augmented_prompt)
        return response.text
    

    # Create a chatbot in order to start the conversation
    # Intialize a chat create history if not created

    if 'history' not in st.session_state:
        st.session_state.history = []
        

    # Display the History 
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.write(":green[User:] :blue[{}]".format(msg["text"]))
        else:
            st.write(":orange[ChatBot:] :blue[{}]".format(msg["text"]))

    

    # Input from the user using streamlit form
    with st.form('chatbot form',clear_on_submit=True):
        user_query = st.text_area('Ask Anything:')
        send = st.form_submit_button('Send')
        

    # Start the conversation and append output and query in history

    if user_query and send:
        st.session_state.history.append({'role':'user','text':user_query})
        st.session_state.history.append({'role':'ChatBot','text': gen_content(user_query)})
        st.rerun()

