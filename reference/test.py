import os
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from mcp import MCPServer, MCPClient, ToolCall, AgentCall
from langgraph.graph import END, StateGraph
import httpx
import openai

# === Config ===
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# === App and MCP Server ===
app = FastAPI()
mcp_server = MCPServer()
app.include_router(mcp_server.router)

# === MCP Client (loopback) ===
mcp_client = MCPClient(base_url="http://localhost:8000")

# === Real Tools ===

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

# === GPT Agents using MCPClient ===

async def eq_agent(input: dict) -> dict:
    news = await mcp_client.call_tool(ToolCall(tool="live_news_tool", input={"topic": input.get("topic", "AI")}))
    return {"eq_response": f"EQ analysis based on: {news['news']}"}

async def fi_agent(input: dict) -> dict:
    price = await mcp_client.call_tool(ToolCall(tool="index_price_tool", input={"index": input.get("index", "SPX")}))
    return {"fi_response": f"FI analysis based on: {price['index_price']}"}

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

# === LangGraph Orchestration ===

@app.post("/langgraph_orchestrate")
async def langgraph_orchestrate(query: str):
    async def fetch_news(state):
        result = await mcp_client.call_tool(ToolCall(tool="live_news_tool", input={"topic": query}))
        return {**state, **result}

    async def fetch_index(state):
        result = await mcp_client.call_tool(ToolCall(tool="index_price_tool", input={"index": "NDX"}))
        return {**state, **result}

    async def run_eq(state):
        result = await mcp_client.call_agent(AgentCall(agent="eq_agent", input=state))
        return {**state, **result}

    async def run_fi(state):
        result = await mcp_client.call_agent(AgentCall(agent="fi_agent", input=state))
        return {**state, **result}

    async def summarize(state):
        messages = [
            {"role": "system", "content": "Summarize the following financial insights for the user."},
            {"role": "user", "content": f"EQ: {state.get('eq_response', '')}\nFI: {state.get('fi_response', '')}"}
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=messages
        )
        return {**state, "summary": response.choices[0].message.content.strip()}

    graph = StateGraph()
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("fetch_index", fetch_index)
    graph.add_node("eq_agent", run_eq)
    graph.add_node("fi_agent", run_fi)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("fetch_news")
    graph.add_edge("fetch_news", "fetch_index")
    graph.add_edge("fetch_index", ["eq_agent", "fi_agent"])
    graph.add_edge("eq_agent", "summarize")
    graph.add_edge("fi_agent", "summarize")
    graph.add_edge("summarize", END)

    compiled = graph.compile()
    result = await compiled.invoke({})
    return result

# === Autonomous Orchestration ===

@app.post("/autonomous_orchestrate")
async def autonomous_orchestrate(query: str):
    selected_agents = await gpt_orchestrator(query)
    state = {"topic": query, "index": "NDX"}
    results = {}
    for agent in selected_agents:
        result = await mcp_client.call_agent(AgentCall(agent=agent, input=state))
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

# === Register with MCP Server ===

@app.on_event("startup")
async def startup():
    await mcp_server.register_tool("live_news_tool", live_news_tool)
    await mcp_server.register_tool("index_price_tool", index_price_tool)
    await mcp_server.register_agent("eq_agent", eq_agent)
    await mcp_server.register_agent("fi_agent", fi_agent)