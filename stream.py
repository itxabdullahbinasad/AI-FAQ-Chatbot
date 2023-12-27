import streamlit as st
from main import create_vector_db, get_qa_chain

st.title("Codnanics Chatbot")
    # Create Knowledgebase button
btn=st.button("Create KnowledgeBase 🐱‍👤")

if btn :
    pass 
question = st.text_input("Question:")
with st.spinner("Please Wait 🐱‍🏍..."):
    if question:
            chain = get_qa_chain()
            response = chain(question)

            st.header("Answer:")
            st.write(response["result"])
