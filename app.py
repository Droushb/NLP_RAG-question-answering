import streamlit as st
from config import CONFIG
from model.main import process_query

st.title("RAG Question Answering System")

# Instructions
st.write("""
Welcome to the Retrieval-Augmented Generation (RAG) Question Answering System.

### What does this system do?
- Searches through a collection of the first 50,000 documents of the dataset to find the most relevant information based on your question using **BM25** and **Semantic Search**.
- Generates accurate answers using the retrieved documents with the power of **OpenAI API GPT-4o-mini**.
- Provides citations for every piece of information to ensure transparency and trustworthiness.

### Instructions
1. **Enter your OpenAI API Key**: You can use your own key.
2. **Ask Your Question**: Type your question in the input box.
3. **Choose a Retrieval Method**:
   - **BM25**: A keyword-based retrieval method.
   - **Semantic Search**: A context-based retrieval method powered by embeddings.
4. **Generate the Answer**: Click the "Generate Answer" button to retrieve relevant documents and generate a detailed answer.

Feel free to experiment with different questions and retrieval methods to explore how the system performs!
""")

llm_key = st.text_input("Enter your LLM API Key", type="password")
# if st.checkbox("Use Test API Key"):
#     llm_key = CONFIG['LLM_API_key']
if not llm_key:
    st.warning("Please provide your LLM API Key to proceed.")
    st.stop()

query = st.text_input("Enter your question")
retrieval_method = st.radio(
    "Select Retrieval Method",
    ("BM25", "Semantic Search")
)

if st.button("Generate Answear"):
    if not query.strip():
        st.warning("Please enter a question to process.")
    else:
        with st.spinner("Processing your query..."):
            try:
                retrieved_docs, answer = process_query(llm_key, query, retrieval_method)
                
                st.subheader("Retrieved Documents")
                for doc in retrieved_docs:
                    st.write(f"- {doc}")
                
                st.subheader("Generated Answer")
                st.text_area("Generated Answer", value=answer, height=CONFIG['TEXTAREA_HEIGHT'], disabled=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown(
    """
    <style>
    .stTextArea {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)
