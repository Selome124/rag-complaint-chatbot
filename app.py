import streamlit as st

st.set_page_config(page_title="Complaint Analysis Chatbot")

st.title("📊 Intelligent Complaint Analysis")
st.write("RAG-powered chatbot for financial complaints")

question = st.text_input("Ask a question about customer complaints:")

if st.button("Ask"):
    st.info("RAG pipeline not connected yet.")
