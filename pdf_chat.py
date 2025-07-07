
#!pip install streamlit openai langchain faiss-cpu sentence-transformers pypdf

#!pip install PyPDF2

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import pickle

# Page config
st.set_page_config(
    page_title="RAG PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGSystem:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", vector_db_path="./vector_db"):
        self.embedding_model_name = embedding_model
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)

        # Initialize components
        self.embedding_model = None
        self.dimension = 384
        self.index = None
        self.chunks = []
        self.metadata = []

        # Initialize OpenAI client
        self.client = None
        if os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    @st.cache_resource
    def load_embedding_model(_self):
        return SentenceTransformer(_self.embedding_model_name)

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        chunks = self.text_splitter.split_text(text)
        return chunks

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if self.embedding_model is None:
            self.embedding_model = self.load_embedding_model()
        embeddings = self.embedding_model.encode(texts)
        return embeddings

    def build_vector_database(self, pdf_file, filename: str):
        """Build vector database from PDF"""
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_file)
        if not text:
            st.error("No text extracted from PDF")
            return False

        # Chunk the text
        chunks = self.chunk_text(text)

        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = self.create_embeddings(chunks)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))

        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = [{"chunk_id": i, "source": filename} for i in range(len(chunks))]

        # Save vector database locally
        self.save_vector_database()

        return True

    def save_vector_database(self):
        """Save vector database to local storage"""
        # Save FAISS index
        faiss.write_index(self.index, str(self.vector_db_path / "faiss_index.idx"))

        # Save chunks and metadata
        with open(self.vector_db_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        with open(self.vector_db_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load_vector_database(self):
        """Load vector database from local storage"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.vector_db_path / "faiss_index.idx"))

            # Load chunks and metadata
            with open(self.vector_db_path / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)

            with open(self.vector_db_path / "metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)

            return True
        except Exception as e:
            return False

    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks based on query"""
        if self.index is None:
            return []

        if self.embedding_model is None:
            self.embedding_model = self.load_embedding_model()

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Return relevant chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })

        return results

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using LLM with retrieved context"""
        if not self.client:
            return "Please set your OpenAI API key in the sidebar to generate answers."

        # Combine context chunks
        context = "\n\n".join(context_chunks)

        # Create prompt
        prompt = f"""
        Based on the following context, answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {query}

        Answer:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"

    def answer_question(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using RAG"""
        if self.index is None:
            if not self.load_vector_database():
                return {"error": "No vector database available. Please upload and process a PDF first."}

        # Search for relevant chunks
        similar_chunks = self.search_similar_chunks(query, top_k)

        if not similar_chunks:
            return {"answer": "No relevant information found"}

        # Extract context chunks
        context_chunks = [chunk["chunk"] for chunk in similar_chunks]

        # Generate answer
        answer = self.generate_answer(query, context_chunks)

        return {
            "answer": answer,
            "sources": similar_chunks,
            "context_used": len(context_chunks)
        }

@st.cache_resource
def get_rag_system():
    return RAGSystem()

# Main UI
def main():
    st.title("📚 RAG PDF Q&A System")
    st.markdown("Upload a PDF document and ask questions about its content!")

    # Initialize RAG system
    rag_system = get_rag_system()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            rag_system.client = OpenAI(api_key=api_key)

        st.divider()

        # Model settings
        st.subheader("Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
            help="Choose the embedding model for text similarity"
        )

        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="Size of text chunks")
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, help="Overlap between chunks")

        # Update text splitter if changed
        if chunk_size != 1000 or chunk_overlap != 200:
            rag_system.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )

        st.divider()

        # Database status
        st.subheader("Database Status")
        if rag_system.load_vector_database():
            st.success(f"✅ Database loaded ({len(rag_system.chunks)} chunks)")
            if st.button("Clear Database"):
                # Clear database files
                for file in rag_system.vector_db_path.glob("*"):
                    file.unlink()
                st.rerun()
        else:
            st.warning("⚠️ No database found")

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📄 Document Upload")

        # File upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")

            # Display file info
            st.write(f"File size: {uploaded_file.size} bytes")

            # Process button
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    progress_bar = st.progress(0)

                    # Update progress
                    progress_bar.progress(25)
                    time.sleep(0.5)

                    # Build vector database
                    success = rag_system.build_vector_database(uploaded_file, uploaded_file.name)

                    progress_bar.progress(100)

                    if success:
                        st.success(f"✅ Document processed successfully!")
                        st.success(f"Created {len(rag_system.chunks)} text chunks")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process document")

    with col2:
        st.header("💬 Ask Questions")

        # Check if database is available
        if not rag_system.load_vector_database():
            st.info("👆 Please upload and process a PDF document first")
            return

        # Question input
        question = st.text_input("Enter your question:", placeholder="What is this document about?")

        # Advanced options
        with st.expander("Advanced Options"):
            top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
            show_sources = st.checkbox("Show source chunks", True)

        # Answer button
        if st.button("🔍 Get Answer", type="primary") and question:
            with st.spinner("Searching for answer..."):
                result = rag_system.answer_question(question, top_k)

                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display answer
                    st.subheader("Answer:")
                    st.write(result["answer"])

                    # Display sources if requested
                    if show_sources and "sources" in result:
                        st.subheader("📖 Source Chunks:")
                        for i, source in enumerate(result["sources"]):
                            with st.expander(f"Chunk {i+1} (Score: {source['score']:.4f})"):
                                st.write(source["chunk"])

                    # Display stats
                    if "context_used" in result:
                        st.info(f"Used {result['context_used']} chunks for context")

        # Chat history (simple implementation)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💭 Recent Questions")
            for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {q[:50]}..."):
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {a}")

        # Add to chat history when answer is generated
        if st.button("🔍 Get Answer", type="primary") and question:
            if "answer" in result:
                st.session_state.chat_history.append((question, result["answer"]))

if __name__ == "__main__":
    main()

