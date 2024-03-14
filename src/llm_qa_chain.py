from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


class LangchainRunner:

    def __init__(self, retriever,
                 hub_prompt_template: str = "rlm/rag-prompt",
                 llm_model_name: str = "gpt-4-turbo-preview",
                 temperature: float = 0):

        self.prompt = self._load_prompt_from_hub(hub_prompt_template)
        self.llm = self._load_llm_model(llm_model_name, temperature)
        self.rag_chain = self._format_chain(retriever, self.prompt, self.llm)

    @staticmethod
    def _load_prompt_from_hub(hub_prompt_template: str):
        assert hub_prompt_template is not None, "hub_prompt_template must be provided."
        return hub.pull(hub_prompt_template)

    @staticmethod
    def _load_llm_model(llm_model_name: str, temperature: float):
        assert llm_model_name is not None, "llm_model_name must be provided."
        return ChatOpenAI(model_name=llm_model_name, temperature=temperature)

    def _format_chain(self, retriever, prompt, llm):

        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        return rag_chain

    def invoke(self, input: str):
        return self.rag_chain.invoke(input)
