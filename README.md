# 💬 Multi-Feature AI Chatbot (LangGraph + Gemini)

A modular, multi-tier AI Chatbot built using **LangGraph**, **Google Gemini**, and **Streamlit**. It supports PDF document retrieval (RAG), external tool calling (Web Search, Calculator, Stock Price API), and full conversation history persistence (Resume Chat) using SQLite.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> **Live Application URL:** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(Replace with your deployed Streamlit Community Cloud link)*

---

## ✨ Features

- **📄 Document RAG (PDF Q&A)**: Upload PDFs, chunk & embed with Google Gemini Embeddings, and query with FAISS vector search.
- **🔧 Multi-Tool Function Calling**: Web search (DuckDuckGo), mathematical calculations, and real-time stock price lookup.
- **💾 Conversation Persistence (Resume Chat)**: Multi-threaded chat history stored in SQLite checkpointer (`chatbot.db`) allowing you to resume previous sessions anytime.
- **⚡ Real-Time Streaming**: Token-by-token response streaming with interactive status containers.
- **🤖 Powered by Google Gemini**: Uses `gemini-3.1-flash-lite` and `models/gemini-embedding-001`.

---

## 🛠️ Tech Stack

- **Framework:** [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain)
- **LLM & Embeddings:** [Google Gemini API](https://aistudio.google.com/) (`langchain-google-genai`)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Checkpointer / Persistence:** SQLite (`langgraph-checkpoint-sqlite`)

---

## ⚡ Quick Start

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create & activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Run the Application
Run your desired chatbot version:

- **PDF RAG & Multi-Utility Chatbot (Full Feature):**
  ```bash
  streamlit run streamlit_rag_frontend.py
  ```

- **Tool Calling Chatbot (Search, Calculator, Stock):**
  ```bash
  streamlit run streamlit_frontend_tool.py
  ```

- **Resume Chat / Database Persistent Chatbot:**
  ```bash
  streamlit run streamlit_frontend_database.py
  ```

---

## ☁️ Deployment on Streamlit Cloud

1. Push your code to GitHub (ensure `.env` and `venv` are in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set the **Main file path** to `streamlit_rag_frontend.py` (or your preferred frontend).
4. Under **Advanced settings > Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_google_gemini_api_key_here"
   ```
5. Click **Deploy**! 🚀
