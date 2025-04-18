from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import asyncio
from mcp_client import MCPClient
import sys
from sse_starlette.sse import EventSourceResponse
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Server API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client and notification subscribers
client = None
notification_subscribers = set()

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    global client
    if len(sys.argv) < 2:
        raise Exception("Server script path not provided")
    client = MCPClient()
    await client.connect_to_server(sys.argv[1])

@app.on_event("shutdown")
async def shutdown_event():
    if client:
        await client.cleanup()

async def notify_subscribers(message: str):
    """Notify all subscribers with a message"""
    for subscriber in notification_subscribers:
        try:
            await subscriber.put({"data": json.dumps({"message": message})})
        except Exception as e:
            print(f"Error notifying subscriber: {e}")

async def generate_events(query: str):
    try:
        # Simulate streaming response by breaking the response into chunks
        response = await client.process_query(query)
        chunks = response.split('\n')
        
        for chunk in chunks:
            if chunk.strip():
                yield {
                    "event": "message",
                    "data": json.dumps({"chunk": chunk})
                }
                await asyncio.sleep(0.1)  # Simulate processing time
        
        # Send notification when query is complete
        await notify_subscribers(f"Query completed: {query[:50]}...")
        
    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
        }
        await notify_subscribers(f"Error processing query: {str(e)}")

@app.post("/query")
async def process_query(request: QueryRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Client not initialized")
    return EventSourceResponse(generate_events(request.query))

@app.get("/notifications")
async def notifications():
    async def event_generator():
        queue = asyncio.Queue()
        notification_subscribers.add(queue)
        try:
            while True:
                data = await queue.get()
                yield data
        finally:
            notification_subscribers.remove(queue)
    
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 