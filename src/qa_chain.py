import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Use an instruction-following prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the following medical question using the provided context. Be concise and only include what the disease is, and how it can be treated or prevented.

Context: {context}

Question: {question}
Answer:
"""
)

def build_qa_chain(vectorstore):
    model_name = os.getenv("LOCAL_MODEL_NAME", "google/flan-t5-base")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7
    )

    local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

    return RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
