import os
from dotenv import load_dotenv
from argparse import ArgumentParser
from vector_index_utils import VectorIndexHandler

load_dotenv()

def main(data_dir, index_save_dir, chunk_size, chunk_overlap):
    vector_index_handler = VectorIndexHandler(data_dir, chunk_size, chunk_overlap)
    docs = vector_index_handler._load_data(data_dir)
    splits = vector_index_handler._split_text(docs)
    vector_index = vector_index_handler._generate_faiss_vector_index(splits)
    vector_index_handler.save_vector_index(index_save_dir, vector_index)
    print("FAISS vector index saved to: {}".format(index_save_dir))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--index-save-dir', type=str, default="./vector_store_index/faiss_index/")
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--chunk-overlap', type=int, default=50)

    args = parser.parse_args()

    main(args.data_dir, args.index_save_dir, args.chunk_size, args.chunk_overlap)
