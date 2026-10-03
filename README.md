# 💬 Multi-Feature AI Chatbot (LangGraph + Gemini)

A modular, multi-tier AI Chatbot built using **LangGraph**, **Google Gemini**, and **Streamlit**. It supports PDF document retrieval (RAG), external tool calling (Web Search, Calculator, Stock Price API), and full conversation history persistence (Resume Chat) using SQLite.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> **Live Application URL:** [https://your-app-name.streamlit.app](https://rag-based-chatbot-htwznbbtsappf5jbmlrnbwu.streamlit.app/)
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
