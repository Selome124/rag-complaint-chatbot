from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from langchain.chains import LLMChain

def load_vector_store():
    """
    Load the pre-built FAISS vector store from disk
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        folder_path="../data/vector_store",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def retrieve_context(question, k=5):
    """
    Retrieve top-k relevant documents for the given question
    """
    vector_store = load_vector_store()

    docs = vector_store.similarity_search(
        question,
        k=k
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    return context, docs


PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Use ONLY the information provided in the context below.
If the context does not contain enough information, say:
"I do not have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""


def load_llm():
    """
    Load HuggingFace LLM using LangChain wrapper
    """
    generator = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=300,
        temperature=0.2
    )

    llm = HuggingFacePipeline(pipeline=generator)
    return llm


def rag_answer(question):
    """
    Full RAG pipeline:
    1. Retrieve context
    2. Format prompt
    3. Generate answer
    """
    context, docs = retrieve_context(question)

    llm = load_llm()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = prompt | llm

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, docs
