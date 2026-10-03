from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:
    MultiServerMCPClient = None
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import os
import sys

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


import streamlit as st

# -------------------
# 1. LLM
# -------------------
def _get_api_key():
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None

api_key = _get_api_key()
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

llm_kwargs = {"model": model_name}
if api_key:
    llm_kwargs["google_api_key"] = api_key

llm = ChatGoogleGenerativeAI(**llm_kwargs)

# -------------------
# 2. Tools
# -------------------
try:
    search_tool = DuckDuckGoSearchRun(region="us-en")
except Exception:
    @tool
    def search_tool(query: str) -> str:
        """Search the web for information."""
        return f"Search result for {query}: Search tool is currently initializing or unavailable."


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "C9PE94QUEW9VWGFM")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={av_key}"
    r = requests.get(url)
    return r.json()


# MultiServerMCPClient configured with Windows support & dynamic paths
mcp_servers = {
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
    }
}

# Add local math server if specified or available
math_server_script = os.getenv("MCP_MATH_SERVER_PATH", "")
if math_server_script and os.path.exists(math_server_script):
    mcp_servers["arith"] = {
        "transport": "stdio",
        "command": sys.executable,  # Use current Python environment on Windows/Linux
        "args": [math_server_script],
    }

client = MultiServerMCPClient(mcp_servers) if (MultiServerMCPClient is not None and mcp_servers) else None


def load_mcp_tools() -> list[BaseTool]:
    if not client:
        return []
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]

    # Check if the most recent message is a ToolMessage
    if messages and isinstance(messages[-1], ToolMessage):
        tool_result = messages[-1].content
        user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        user_query = user_msg.content if user_msg else "the user's question"

        prompt = (
            f"Here is the result from the tool:\n{tool_result}\n\n"
            f"Based on this tool result, provide a helpful and direct answer to: {user_query}"
        )
        response = await llm.ainvoke(
            [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)],
            config=config,
        )
        return {"messages": [response]}

    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
