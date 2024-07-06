import streamlit as st
import time
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User:", message.content)
        else:
            st.write("Bot:", message.content)

def main():
    st.set_page_config(
        page_title="Rag chatbot testing",
        page_icon="🤖",
        layout="wide"
    )
    st.header("Retrieval System implementation")

    user_question = st.text_input("Ask Me a Question from your PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    if user_question:
        if st.session_state.conversation is not None:
            user_input(user_question)
        else:
            st.warning("Please upload and process a PDF file first.")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF file and click on submit and process button", type=['pdf'])

        if st.button("Submit and Process"):
            if pdf_docs is not None:
                with st.spinner('Let me go through it.....'):
                    # Save the uploaded file temporarily to read it
                    with open("temp.pdf", "wb") as f:
                        f.write(pdf_docs.read())
                    
                    time.sleep(15)  # Ensure the spinner is displayed
                    
                    raw_text = get_pdf_text("temp.pdf")
                    chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)

                    st.success("Done")

if __name__ == "__main__":
    main()
