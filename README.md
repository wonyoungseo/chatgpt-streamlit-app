
# OpenAI ChatGPT Streamlit App

> Streamlit application at its simplest form, powered with OpenAI ChatGPT API 

## Prerequisite

This application is built and tested under Python3.9

1. Install dependencies.
   - `pip install -r requirements.txt`

2. Place your text data(`.txt`) in `./data` directory.

3. Prepare `.env` with your own API keys.
    ```
    OPENAI_API_KEY=<YOUR OPENAI API KEY>
    LANGCHAIN_API_KEY=<YOUR LANGCHAIN API KEY>
    ```

4. Create and save vector index.
   - `python src/generate_vector_store_index.py --data-dir="./data"`


## How to run

- `streamlit run app.py`

* * *

## Reference

- [Langchain Tutorial - Q&A with RAG](https://python.langchain.com/docs/use_cases/question_answering/quickstart)