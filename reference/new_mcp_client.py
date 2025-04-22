import asyncio
import json
import os
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

import openai
import tiktoken
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper to count tokens in messages for trimming

def count_tokens(messages: List[Dict[str, str]], model: str = "gpt-4-1106-preview") -> int:
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for m in messages:
        total += len(enc.encode(m.get("content", ""))) + 4  # per-message overhead
    return total

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Single system prompt + conversation history
        self.messages: List[Dict[str, Any]] = []
        # Tool definitions for function-calling
        self.tool_definitions: List[Dict[str, Any]] = []

    async def connect_to_server(self, server_script_path: str):
        if not server_script_path.endswith(('.py', '.js')):
            raise ValueError("Server script must be a .py or .js file")

        cmd = "python" if server_script_path.endswith('.py') else "node"
        params = StdioServerParameters(command=cmd, args=[server_script_path], env=None)
        transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        # Fetch and register tools
        resp = await self.session.list_tools()
        self.available_tools = resp.tools
        for tool in self.available_tools:
            self.tool_definitions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": True}
            })

        # Build single system prompt
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in self.available_tools)
        system_prompt = f"""
You are an AI agent that can interact with an MCP server through tools:

{tool_list}

When you need a tool, return a structured function_call. If no tool is needed, give the final answer directly.
"""
        self.messages = [{"role": "system", "content": system_prompt}]
        print("Connected! Tools:", [t.name for t in self.available_tools])

    async def _smart_trim_history(self, keep_turns: int = 2, max_tokens: int = 120_000):
        # If under limit, no action needed
        if count_tokens(self.messages) <= max_tokens:
            return

        # Extract old history except the last few turns
        old_history = self.messages[1:-keep_turns]
        # Build summarization prompt
        summary_prompt = [
            self.messages[0],
            {"role": "user", "content": (
                "Please summarize the following conversation into a few concise points:\n\n" +
                "\n".join(m.get("content", "") for m in old_history)
            )}
        ]
        # Call the model to summarize
        resp = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=summary_prompt
        )
        summary = resp.choices[0].message.content

        # Rebuild messages: keep system prompt, add summary, then recent turns
        self.messages = [
            self.messages[0],
            {"role": "system", "content": f"Conversation summary:\n{summary}"}
        ] + self.messages[-keep_turns:]

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        try:
            result = await self.session.call_tool(name, args)
            return json.dumps(result.content)
        except Exception as e:
            return f"Error executing {name}: {e}"

    async def process_query(self, query: str, iteration_callback=None) -> str:
        # Append user query
        self.messages.append({"role": "user", "content": query})
        # Smart-trim history if needed
        await self._smart_trim_history()

        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            if iteration_callback:
                iteration_callback(f"Iteration {iteration+1}/{max_iterations}")

            # Let the model decide on a tool or final answer
            resp = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=self.messages,
                functions=self.tool_definitions,
                function_call="auto"
            )
            msg = resp.choices[0].message

            # If model requests a function call, execute it
            if msg.get("function_call"):
                name = msg.function_call.name
                args = json.loads(msg.function_call.arguments or "{}")
                # Record the function call
                self.messages.append(msg.to_dict())

                # Execute tool and record result
                tool_output = await self._execute_tool(name, args)
                self.messages.append({"role": "assistant", "content": tool_output})

                iteration += 1
                continue

            # No function call: return final answer
            answer = msg.content
            self.messages.append({"role": "assistant", "content": answer})
            if iteration_callback:
                iteration_callback("Final answer generated")
            return answer

        # If loop ends without final answer
        return "Maximum iterations reached. Please try again with a more specific query."

    async def chat_loop(self):
        print("MCP Client Ready! (type 'quit' to exit)")
        while True:
            query = input("Query: ").strip()
            if query.lower() == 'quit':
                break
            print(await self.process_query(query))

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_script>")
        return
    client = MCPClient()
    await client.connect_to_server(sys.argv[1])
    try:
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
