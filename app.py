import streamlit as st
from retriever import load_retriever
from llm import generate_answer

st.set_page_config(page_title="Bangla News Q&A Bot", page_icon="📰")

st.title("📰 Bangla News Q&A Bot (RAG)")

# load retriever once
retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("সংবাদ সম্পর্কে প্রশ্ন করুন...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # retrieve docs
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    answer = generate_answer(context, question)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })