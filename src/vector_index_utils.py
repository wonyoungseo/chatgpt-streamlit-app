import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class VectorIndexHandler:

    def __init__(self):

        assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY must be provided. Please set it in the .env file"

    @staticmethod
    def _load_data(data_dir):
        assert data_dir is not None, "data_dir must be set when initializing VectorIndexHandler"

        dir_loader = DirectoryLoader(data_dir)
        return dir_loader.load()

    @staticmethod
    def _split_text(documents, chunk_size: int = 1000, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return text_splitter.split_documents(documents)

    @staticmethod
    def _generate_faiss_vector_index(splits):
        vector_index = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        return vector_index

    @staticmethod
    def _get_retriever(vector_index):
        return vector_index.as_retriever()

    @staticmethod
    def save_vector_index(index_save_path: str, vector_index):
        assert index_save_path is not None, "index_save_path must be provided."
        vector_index.save_local(index_save_path)

    @staticmethod
    def load_vector_index_retriever(index_path: str):
        vector_index = FAISS.load_local(index_path, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        return vector_index.as_retriever()
