import asyncio
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import sys
import json
import openai
from dotenv import load_dotenv
import os
import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.conversation_history = []
        
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.available_tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.available_tools])
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI agent"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.available_tools
        ])
        
        return f"""You are an AI agent that can interact with a database through an MCP server.
You have access to the following tools:

{tool_descriptions}

When responding:
1. Analyze the user's query to determine which tool(s) to use
2. If you need more information, use the appropriate tools to gather it
3. Explain your thought process and the tools you're using
4. Format your response in a clear and helpful way

When you need to use a tool, format your response like this:
TOOL: tool_name
ARGS: {{"arg1": "value1", "arg2": "value2"}}

If you don't need to use any tools, just provide your response directly."""
        
    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI's chat completions API"""
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            # Get response from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()}
                ] + self.conversation_history
            )
            
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Check if the response contains a tool call
            tool_call = self._parse_tool_call(assistant_response)
            if not tool_call:
                # No tool call found, this is the final answer
                return assistant_response
            
            # Execute the tool
            tool_name, args = tool_call
            tool_result = await self._execute_tool(tool_name, args)
            
            # Add tool result to conversation history
            self.conversation_history.append({
                "role": "system",
                "content": f"Tool {tool_name} returned:\n{tool_result}"
            })
            
            iteration += 1
        
        return "Maximum iterations reached. Please try again with a more specific query."
        
    def _parse_tool_call(self, response: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Parse a tool call from the assistant's response"""
        import re
        
        tool_match = re.search(r"TOOL: (\w+)", response)
        args_match = re.search(r"ARGS: ({.*})", response)
        
        if tool_match and args_match:
            tool_name = tool_match.group(1)
            try:
                args = json.loads(args_match.group(1))
                return tool_name, args
            except json.JSONDecodeError:
                return None
        return None
        
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return its result as a string"""
        try:
            result = await self.session.call_tool(tool_name, args)
            return json.dumps(serialize_content(result.content), indent=2)
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
        
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")
                
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

def serialize_content(content):
    if isinstance(content, types.TextContent):
        return content.text
    elif isinstance(content, dict):
        return {k: serialize_content(v) for k, v in content.items()}
    elif isinstance(content, list):
        return [serialize_content(item) for item in content]
    else:
        return content

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
