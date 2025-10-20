# streamlit_app.py
import os
import streamlit as st
from rag_utils import Ingestor
from gemini_client import GeminiClient

st.set_page_config(page_title="RAG + Gemini Demo", layout="wide")

# Initialize ingestor (handles text, chunks, embeddings, etc.)
ingestor = Ingestor()
ingestor.create_or_load_index()

# Try to initialize Gemini (optional)
GEMINI = None
if os.getenv("GEMINI_API_KEY"):
    try:
        GEMINI = GeminiClient()
    except Exception as e:
        st.sidebar.warning(f"Gemini not initialized: {e}")

# Sidebar – Upload or URL
st.sidebar.header("📂 Add your documents")
uploaded = st.sidebar.file_uploader("Upload a PDF / DOCX / TXT file", type=["pdf", "docx", "txt"])
url = st.sidebar.text_input("Or paste a webpage URL")
if st.sidebar.button("Ingest"):
    if not uploaded and not url:
        st.sidebar.warning("Please upload a file or enter a URL.")
    else:
        with st.spinner("Extracting and embedding..."):
            if uploaded:
                fname = uploaded.name
                path = os.path.join("data/docs", fname)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())

                if fname.lower().endswith(".pdf"):
                    text = ingestor.extract_text_from_pdf(path)
                elif fname.lower().endswith(".docx"):
                    text = ingestor.extract_text_from_docx(path)
                else:
                    text = uploaded.getvalue().decode("utf-8")

                chunks = ingestor.chunk_text(text)
                metas = [{"source": fname, "text": c} for c in chunks]
                ingestor.add_texts(chunks, metas)
                st.sidebar.success(f"Ingested {len(chunks)} chunks from {fname}")
            elif url:
                text = ingestor.fetch_url_text(url)
                chunks = ingestor.chunk_text(text)
                metas = [{"source": url, "text": c} for c in chunks]
                ingestor.add_texts(chunks, metas)
                st.sidebar.success(f"Ingested {len(chunks)} chunks from URL")

# Main chat area
st.title("💬 RAG-based Q&A with Gemini")
query = st.text_input("Ask a question about your documents:")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    elif ingestor.index is None or ingestor.index.ntotal == 0:
        st.warning("No documents have been ingested yet.")
    else:
        with st.spinner("Retrieving relevant context..."):
            results = ingestor.query(query, k=5)
            context = "\n---\n".join([r["text"] for r in results])

        prompt = f"""
        You are an assistant. Use the provided context to answer the question.
        If the answer isn't in the context, say "I don't know."

        Context:
        {context}

        Question: {query}
        Answer:
        """

        if GEMINI is None:
            st.info("⚠️ Gemini API key not found. Showing retrieved context instead.")
            st.subheader("Top relevant document chunks:")
            for i, r in enumerate(results, start=1):
                st.markdown(f"**Source {i}: {r['source']}**")
                st.write(r["text"][:1000] + ("..." if len(r["text"]) > 1000 else ""))
        else:
            with st.spinner("Querying Gemini..."):
                answer = GEMINI.generate(prompt)
            st.subheader("🧠 Gemini Answer")
            st.write(answer)
            st.markdown("---")
            st.subheader("Retrieved Sources")
            for i, r in enumerate(results, start=1):
                st.markdown(f"**Source {i}: {r['source']}**")
                st.write(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))
