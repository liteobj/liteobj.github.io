import os
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from mcp.core import MCPRegistry
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
import httpx
import openai

# === Config ===
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# === App and Registry ===
app = FastAPI()
mcp_registry = MCPRegistry()

# === Live Tools ===

async def live_news_tool(input: dict) -> dict:
    topic = input.get("topic", "market")
    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize=1&apiKey={NEWS_API_KEY}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        articles = resp.json().get("articles", [])
        headline = articles[0]["title"] if articles else "No news found."
        return {"news": headline}

async def index_price_tool(input: dict) -> dict:
    index = input.get("index", "SPX")
    url = f"https://api.polygon.io/v2/aggs/ticker/I:{index}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        results = resp.json().get("results", [])
        if results:
            close_price = results[0]["c"]
            return {"index_price": f"{index} closed at {close_price}"}
        return {"index_price": "Price not available."}

# === GPT-Powered Agent ===

async def gpt_agent(agent_name: str, context: dict) -> dict:
    tools_available = {
        "eq_agent": ["live_news_tool"],
        "fi_agent": ["index_price_tool"]
    }

    tools_prompt = {
        "live_news_tool": "fetch live news headlines on a topic",
        "index_price_tool": "get current index/market price"
    }

    tool_options = tools_available.get(agent_name, [])
    tool_descriptions = "\n".join([f"- {tool}: {tools_prompt[tool]}" for tool in tool_options])

    messages = [
        {"role": "system", "content": f"You are the '{agent_name}'. Decide which tools to use:\n{tool_descriptions}"},
        {"role": "user", "content": f"Context: {context}"}
    ]

    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=messages,
        temperature=0.3
    )
    decision = response.choices[0].message.content.lower()
    result = {}

    if "live_news_tool" in decision and "topic" in context:
        news = await mcp_registry.tools["live_news_tool"]({"topic": context["topic"]})
        result.update(news)

    if "index_price_tool" in decision and "index" in context:
        price = await mcp_registry.tools["index_price_tool"]({"index": context["index"]})
        result.update(price)

    result["agent_response"] = f"{agent_name} used tools based on decision: {decision}"
    return result

# === GPT Orchestrator ===

async def gpt_orchestrator(user_query: str) -> list:
    messages = [
        {"role": "system", "content": "Given a user query, choose one or more relevant agents from: ['eq_agent', 'fi_agent']."},
        {"role": "user", "content": f"User query: {user_query}"}
    ]

    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=messages,
        temperature=0.3
    )
    agent_list = response.choices[0].message.content.strip()
    return [agent.strip() for agent in agent_list.replace("[", "").replace("]", "").replace("'", "").split(",")]

# === LangGraph Router ===

@app.post("/langgraph_router")
async def langgraph_router(query: str):
    async def gpt_router(state: dict) -> str:
        messages = [
            {"role": "system", "content": "You are a router. Choose one of ['live_news_tool', 'index_price_tool'] based on user query."},
            {"role": "user", "content": f"User Query: {state['query']}"}
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=messages,
            tool_choice="auto",
            tools=[
                {"type": "function", "function": {
                    "name": "live_news_tool",
                    "description": "Fetch news based on topic",
                    "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
                }},
                {"type": "function", "function": {
                    "name": "index_price_tool",
                    "description": "Get index price for a symbol",
                    "parameters": {"type": "object", "properties": {"index": {"type": "string"}}, "required": ["index"]}
                }}
            ]
        )
        return response.choices[0].message.tool_calls[0].function.name

    graph = StateGraph()
    graph.add_node("router", gpt_router)
    graph.add_node("live_news_tool", ToolNode(tool=live_news_tool))
    graph.add_node("index_price_tool", ToolNode(tool=index_price_tool))
    graph.set_entry_point("router")
    graph.add_edge("router", "live_news_tool")
    graph.add_edge("router", "index_price_tool")
    graph.add_edge("live_news_tool", END)
    graph.add_edge("index_price_tool", END)
    compiled = graph.compile()
    return await compiled.invoke({"query": query})

# === Autonomous GPT-driven Orchestration ===

@app.post("/autonomous_orchestrate")
async def autonomous_orchestrate(query: str):
    selected_agents = await gpt_orchestrator(query)
    state = {"topic": query, "index": "NDX"}
    results = {}
    for agent in selected_agents:
        result = await gpt_agent(agent, state)
        results[agent] = result

    summary_prompt = [
        {"role": "system", "content": "You summarize results from agents for user clarity."},
        {"role": "user", "content": f"Agent results: {results}"}
    ]
    final = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=summary_prompt
    )
    return {
        "selected_agents": selected_agents,
        "agent_outputs": results,
        "gpt_summary": final.choices[0].message.content.strip()
    }

# === Register Tools on Startup ===

@app.on_event("startup")
async def startup():
    mcp_registry.register_tool("live_news_tool", live_news_tool)
    mcp_registry.register_tool("index_price_tool", index_price_tool)
    mcp_registry.register_agent("eq_agent", lambda ctx: gpt_agent("eq_agent", ctx))
    mcp_registry.register_agent("fi_agent", lambda ctx: gpt_agent("fi_agent", ctx))
