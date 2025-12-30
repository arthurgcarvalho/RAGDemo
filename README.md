# 📚 RAG Pipeline Visualizer

A modern, interactive Streamlit application that demonstrates the end-to-end **Retrieval-Augmented Generation (RAG)** pipeline. This educational tool visualizes how text data is chunked, embedded, indexed, and retrieved using **Google Gemini** and **FAISS**.

![App Screenshot](https://via.placeholder.com/800x450.png?text=RAG+Pipeline+Visualizer)
*(Replace with actual screenshot)*

## ✨ Features

-   **📄 Document Ingestion**: Upload your own text files or use the built-in **U.S. Constitution** sample.
-   **🧠 Smart Chunking**: Uses `LangChain` to intelligently split text while respecting natural boundaries (paragraphs, sentences).
-   **⚡ Fast Embedding**: Implements **batch processing** with Google's `gemini-embedding-001` model for high-speed vector generation.
-   **🔍 Vector Search**: Powered by **FAISS** (Facebook AI Similarity Search) for efficient, low-latency similarity matching.
-   **🎨 Interactive UI**: Clean, tabbed interface built with Streamlit for a seamless user experience.
-   **🔒 Secure**: API keys are managed strictly via environment variables.

## 🛠️ Prerequisites

-   Python 3.8+
-   A [Google Cloud API Key](https://aistudio.google.com/app/apikey) with access to Gemini models.

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/rag-pipeline-visualizer.git
    cd rag-pipeline-visualizer
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY=your_actual_api_key_here
    ```

## 💻 Usage

1.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2.  **Open the App**:
    The application will launch automatically in your browser at `http://localhost:8501`.

3.  **Explore**:
    *   **Tab 1 (Ingestion)**: Click "Process Document" to chunk and index the text. You can adjust the chunk size slider to see how it affects the number of vectors.
    *   **Tab 2 (Retrieval)**: Enter a natural language query (e.g., *"What is the First Amendment?"*) to find the most relevant sections of the document.

## 📂 Project Structure

-   `app.py`: Main Streamlit application file handling the UI and state.
-   `rag_engine.py`: Core logic for text processing, embedding generation, and vector search.
-   `constitution.txt`: Default sample document.
-   `requirements.txt`: Python dependencies.

## 🔧 Technologies Used

-   **Streamlit**: Frontend framework.
-   **Google Gemini API**: Embedding generation (`models/gemini-embedding-001`).
-   **FAISS**: Vector database and similarity search.
-   **LangChain**: Advanced text splitting.
-   **NumPy**: Vector manipulation.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
