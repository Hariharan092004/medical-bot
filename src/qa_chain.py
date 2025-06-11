import os
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question briefly and clearly.
Only include what the disease is and how it can be treated or prevented.

{context}

Question: {question}
Helpful Answer:
"""
)

def build_qa_chain(vectorstore):
    model_name = os.getenv("LOCAL_MODEL_NAME", "distilgpt2")

    hf_pipeline = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

    local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Now use the custom prompt in the chain
    return RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
