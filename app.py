import os
from dotenv import load_dotenv

import streamlit as st

from src.vector_index_utils import VectorIndexHandler
from src.llm_qa_chain import LangchainRunner

load_dotenv()

retriever = VectorIndexHandler().load_vector_index_retriever("./vector_store_index/faiss_index/")
runner = LangchainRunner(retriever=retriever)

def app():

    st.set_page_config(
        page_title="테니스 Q&A GPT"
    )

    st.header("테니스 Q&A GPT")

    input_text = st.text_input(label="테니스와 관련된 질문을 입력하세요.")

    if st.button(label="질문하기"):

        with st.spinner("답변 생성 중 ..."):
            output = runner.invoke(input=input_text)

        st.subheader("Answer:")
        st.markdown("> {}".format(output))




if __name__ == "__main__":
    app()
