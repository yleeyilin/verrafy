import time
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings

def query(question, llm):
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(question)

def normal_chat(llm, documents):
    chain = load_qa_chain(llm, chain_type="stuff")
    embeddings = HuggingFaceHubEmbeddings()
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 0,
    )

    splitDocs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(splitDocs, embeddings)
    st.divider()
    if st.button('Clear session'):
        st.session_state.messages = []

    with st.container():
        # Initialise chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask Anything"):
            st.session_state.messages.append({"role": "user", "content" : "prompt"})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                docs = db.similarity_search(prompt, 1)
                fileCheck = chain.run(input_documents=docs, question=prompt)
                assistant_response = fileCheck
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + " ")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})