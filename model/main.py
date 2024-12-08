import streamlit as st
from model.questionAnsweringBot import QuestionAnsweringBot
from model.retriever import Retriever

def process_query(llm_key, query, retrieval_method):
    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever()
        print("Loading and preparing dataset...")
        st.session_state.retriever.load_and_prepare_dataset()
        st.session_state.retriever.prepare_bm25()
        st.session_state.retriever.compute_embeddings()

    retriever = st.session_state.retriever

    if retrieval_method == "BM25":
        print("Retrieving documents using BM25...")
        retrieved_docs = retriever.retrieve_documents_bm25(query)
    else:
        print("Retrieving documents using Semantic Search...")
        retrieved_docs = retriever.retrieve_documents_semantic(query)

    bot = QuestionAnsweringBot(llm_key)
    prompt = getPrompt(retrieved_docs, query)
    answer = bot.generate_answer(prompt)

    return retrieved_docs, answer

def getPrompt(retrieved_docs, query):
    prompt = (
        "You are an LM integrated into an RAG system that answers questions based on provided documents.\n"
        "Rules:\n"
        "- Reply with the answer only and nothing but the answer.\n"
        "- Say 'I don't know' if you don't know the answer.\n"
        "- Use only the provided documents.\n"
        "- Citations are required. Include the document and chunk number in square brackets after the information (e.g., [Document 1, Chunk 2]).\n\n"
        "Documents:\n"
    )

    for i, doc in enumerate(retrieved_docs):
        prompt += f"Document {i + 1}: {doc}\n"

    prompt += f"\nQuery: {query}\n"

    return prompt
