"""import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import openai
from io import BytesIO

openai.api_key = 'Openai_api_key'  
st.set_page_config(page_title="SOFTDOC")

def process_pdfs(pdf_files):
    raw_text = ''
    for pdf_file in pdf_files:
        pdf_bytes = pdf_file.getvalue()
        pdf_io = BytesIO(pdf_bytes)
        pdfreader = PdfReader(pdf_io)
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    return raw_text

def create_vectorstore(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, model

def find_context(query, index, embeddings, model, texts, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_texts = [texts[i] for i in indices[0]]
    return " ".join(relevant_texts)

def chatbot(prompt, chat_history, context):
    messages = [{"role": "system", "content":You are a helpful assistant designed to aid users in understanding software documentation quickly and easily. 
Your goal is to simplify complex concepts, provide clear explanations, and offer concise answers to questions related to software features, functionalities, and usage. 
Use the provided context to assist users effectively, ensuring they grasp the essential information without unnecessary jargon. 
If users greet you or ask general questions, respond warmly and briefly before addressing their main inquiries. 
Your aim is to be helpful, friendly, and thorough, ensuring users leave with a clear understanding of the software documentation.
}]
    # Limit chat history to the last 5 messages
    recent_history = chat_history[-5:]  # Adjust this number as needed    
    for message in chat_history:
        messages.append({"role": message["role"], "content": message["content"]})
    if context:
        messages.append({"role": "system", "content": f"Context: {context}"})
    messages.append({"role": "user", "content": prompt})
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def history(role, content, chat_history):
    chat_history.append({"role": role, "content": content})
    return chat_history

def main():
    st.title("SOFTDOC")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    
    if uploaded_files:
        raw_text = process_pdfs(uploaded_files)
        st.write("PDFs processed successfully!")

        # Create vector store from the uploaded PDF contents
        all_texts = [raw_text]
        index, embeddings, model = create_vectorstore(all_texts)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You: ", key="input")
        
        if user_input:
            context = find_context(user_input, index, embeddings, model, all_texts) if all_texts else ""
            st.session_state.chat_history = history("user", user_input, st.session_state.chat_history)
            gpt_response = chatbot(user_input, st.session_state.chat_history, context)
            st.session_state.chat_history = history("assistant", gpt_response, st.session_state.chat_history)

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])

if __name__ == "__main__":
    main()"""

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import openai
from io import BytesIO

openai.api_key = 'open_ai_api_key'

st.set_page_config(page_title="SOFTDOC")

def process_pdfs(pdf_files):
    raw_text = ''
    for pdf_file in pdf_files:
        pdf_bytes = pdf_file.getvalue()
        pdf_io = BytesIO(pdf_bytes)
        pdfreader = PdfReader(pdf_io)
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    return raw_text

def create_vectorstore(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, model

def find_context(query, index, embeddings, model, texts, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_texts = [texts[i] for i in indices[0]]
    return " ".join(relevant_texts)

def trim_context_to_fit(context, max_tokens=3000):
    # Simplistic approach to trim long context to fit within the token limit
    # You can use OpenAI's model to summarize if needed
    if len(context.split()) > max_tokens:
        # Trimming the context
        return " ".join(context.split()[:max_tokens]) + "..."
    return context

def chatbot(prompt, chat_history, context):
    messages = [{"role": "system", "content":"""You are a helpful assistant designed to aid users in understanding software documentation quickly and easily. 
Your goal is to simplify complex concepts, provide clear explanations, and offer concise answers to questions related to software features, functionalities, and usage. 
Use the provided context to assist users effectively, ensuring they grasp the essential information without unnecessary jargon. 
If users greet you or ask general questions, respond warmly and briefly before addressing their main inquiries. 
Your aim is to be helpful, friendly, and thorough, ensuring users leave with a clear understanding of the software documentation.
"""}]
    
    # Only add the last 5 messages to stay within the token limit
    recent_history = chat_history[-5:]
    
    for message in recent_history:
        messages.append({"role": message["role"], "content": message["content"]})
    
    if context:
        context = trim_context_to_fit(context)  # Trim context to avoid going over token limit
        messages.append({"role": "system", "content": f"Context: {context}"})
    
    messages.append({"role": "user", "content": prompt})
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    return response.choices[0].message.content.strip()

def history(role, content, chat_history):
    chat_history.append({"role": role, "content": content})
    return chat_history

def main():
    st.title("SOFTDOC")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    
    if uploaded_files:
        raw_text = process_pdfs(uploaded_files)
        st.write("PDFs processed successfully!")

        # Create vector store from the uploaded PDF contents
        all_texts = [raw_text]
        index, embeddings, model = create_vectorstore(all_texts)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You: ", key="input")
        
        if user_input:
            context = find_context(user_input, index, embeddings, model, all_texts) if all_texts else ""
            st.session_state.chat_history = history("user", user_input, st.session_state.chat_history)
            gpt_response = chatbot(user_input, st.session_state.chat_history, context)
            st.session_state.chat_history = history("assistant", gpt_response, st.session_state.chat_history)

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])

if __name__ == "__main__":
    main()

