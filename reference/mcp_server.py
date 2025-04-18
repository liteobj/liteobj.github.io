import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import jaydebeapi
from dotenv import load_dotenv
import os
from mcp import Server, Tool, ToolCallResult, types
from pydantic import AnyUrl

load_dotenv()  # load environment variables from .env

# Configure logging to output to both console and file
logger = logging.getLogger('mcp_sql_server')
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('mcp_server.log')
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Starting MCP SQL Server")

class SQLServer:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.insights = []
        
    async def connect(self):
        """Connect to MS SQL Server using JDBC"""
        server = os.getenv("SQL_SERVER")
        database = os.getenv("SQL_DATABASE")
        username = os.getenv("SQL_USERNAME")
        password = os.getenv("SQL_PASSWORD")
        
        # JDBC connection string
        jdbc_url = f"jdbc:sqlserver://{server};databaseName={database}"
        
        # Path to the SQL Server JDBC driver
        driver_path = os.getenv("JDBC_DRIVER_PATH", "mssql-jdbc-12.4.2.jre11.jar")
        
        # JDBC driver class
        driver_class = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
        
        # Connect using jaydebeapi
        self.connection = jaydebeapi.connect(
            driver_class,
            jdbc_url,
            [username, password],
            driver_path
        )
        self.cursor = self.connection.cursor()
        
    async def close(self):
        """Close the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            
    async def get_table_list(self) -> List[str]:
        """Get list of all tables in the database"""
        self.cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
        """)
        return [row[0] for row in self.cursor.fetchall()]
        
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a specific table"""
        self.cursor.execute("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH,
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
        """, [table_name])
        
        columns = []
        for row in self.cursor.fetchall():
            column = {
                "name": row[0],
                "type": row[1],
                "max_length": row[2],
                "nullable": row[3] == "YES"
            }
            columns.append(column)
            
        return {
            "table_name": table_name,
            "columns": columns
        }
        
    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        try:
            self.cursor.execute(query)
            
            # For SELECT queries, return the results
            if query.strip().upper().startswith("SELECT"):
                columns = [desc[0] for desc in self.cursor.description]
                results = []
                for row in self.cursor.fetchall():
                    result = {}
                    for i, value in enumerate(row):
                        result[columns[i]] = value
                    results.append(result)
                return results
            # For write operations, return affected rows
            else:
                affected = self.cursor.rowcount
                self.connection.commit()
                return [{"affected_rows": affected}]
        except Exception as e:
            logger.error(f"Database error executing query: {e}")
            raise

    def _synthesize_memo(self) -> str:
        """Synthesize the insights memo content"""
        if not self.insights:
            return "No insights recorded yet."
        return "\n\n".join(f"- {insight}" for insight in self.insights)

    def _is_read_only_query(self, query: str) -> bool:
        """Check if a query is read-only by examining the SQL statement"""
        query = query.strip().upper()
        read_only_keywords = {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN"}
        write_keywords = {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "MERGE"}
        
        # Check for write operations first
        for keyword in write_keywords:
            if query.startswith(keyword):
                return False
                
        # Check for read operations
        for keyword in read_only_keywords:
            if query.startswith(keyword):
                return True
                
        # If we can't determine, assume it's a write operation for safety
        return False

class MCPServer:
    def __init__(self):
        self.sql_server = SQLServer()
        self.server = Server("mcp_server")
        self._register_handlers()
        
    def _register_handlers(self):
        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            return [
                types.Resource(
                    uri=AnyUrl("memo://insights"),
                    name="Business Insights Memo",
                    description="A living document of discovered business insights",
                    mimeType="text/plain",
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> str:
            if uri.scheme != "memo":
                raise ValueError(f"Unsupported URI scheme: {uri.scheme}")
            path = str(uri).replace("memo://", "")
            if not path or path != "insights":
                raise ValueError(f"Unknown resource path: {path}")
            return self.sql_server._synthesize_memo()

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[types.Prompt]:
            return [
                types.Prompt(
                    name="mcp-demo",
                    description="A prompt to seed the database with initial data and demonstrate what you can do with an MCP Server",
                    arguments=[
                        types.PromptArgument(
                            name="topic",
                            description="Topic to seed the database with initial data",
                            required=True,
                        )
                    ],
                )
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: Dict[str, str] | None) -> types.GetPromptResult:
            if name != "mcp-demo":
                raise ValueError(f"Unknown prompt: {name}")
            if not arguments or "topic" not in arguments:
                raise ValueError("Missing required argument: topic")
            
            topic = arguments["topic"]
            prompt = f"""
            The assistant's goal is to walk through an informative demo of MCP. To demonstrate the Model Context Protocol (MCP) we will leverage this example server to interact with a SQL Server database.

            The user has provided the topic: {topic}. The goal of the following instructions is to walk the user through the process of using the 3 core aspects of an MCP server. These are: Prompts, Tools, and Resources.

            The user is now ready to begin the demo.
            """
            
            return types.GetPromptResult(
                description=f"Demo template for {topic}",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=prompt.strip()),
                    )
                ],
            )
            
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="read_query",
                    description="Execute a SELECT query on the SQL Server database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SELECT SQL query to execute"},
                        },
                        "required": ["query"],
                    }
                ),
                types.Tool(
                    name="write_query",
                    description="Execute an INSERT, UPDATE, or DELETE query on the SQL Server database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query to execute"},
                        },
                        "required": ["query"],
                    }
                ),
                types.Tool(
                    name="create_table",
                    description="Create a new table in the SQL Server database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "CREATE TABLE SQL statement"},
                        },
                        "required": ["query"],
                    }
                ),
                types.Tool(
                    name="list_tables",
                    description="List all tables in the SQL Server database",
                    inputSchema={}
                ),
                types.Tool(
                    name="describe_table",
                    description="Get the schema information for a specific table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "Name of the table to describe"},
                        },
                        "required": ["table_name"],
                    }
                ),
                types.Tool(
                    name="append_insight",
                    description="Add a business insight to the memo",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "insight": {"type": "string", "description": "Business insight discovered from data analysis"},
                        },
                        "required": ["insight"],
                    }
                )
            ]
           
        @self.server.call_tool()
        async def handle_call_tool(name: str, args: Dict[str, Any]) -> ToolCallResult:
            try:
                if name == "list_tables":
                    result = await self.sql_server.get_table_list()
                elif name == "describe_table":
                    result = await self.sql_server.get_table_schema(args["table_name"])
                elif name == "append_insight":
                    if not args or "insight" not in args:
                        raise ValueError("Missing insight argument")
                    self.sql_server.insights.append(args["insight"])
                    result = "Insight added to memo"
                elif name in ["read_query", "write_query", "create_table"]:
                    if not args or "query" not in args:
                        raise ValueError("Missing query argument")
                    query = args["query"]
                    
                    if name == "read_query" and not query.strip().upper().startswith("SELECT"):
                        raise ValueError("Only SELECT queries are allowed for read_query")
                    elif name == "write_query" and query.strip().upper().startswith("SELECT"):
                        raise ValueError("SELECT queries are not allowed for write_query")
                    elif name == "create_table" and not query.strip().upper().startswith("CREATE TABLE"):
                        raise ValueError("Only CREATE TABLE statements are allowed")
                    
                    result = await self.sql_server.execute_query(query)
                else:
                    return ToolCallResult(
                        success=False,
                        content=f"Unknown tool: {name}"
                    )
                    
                return ToolCallResult(
                    success=True,
                    content=result
                )
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                return ToolCallResult(
                    success=False,
                    content=f"Error executing tool {name}: {str(e)}"
                )

    async def initialize(self):
        """Initialize the server and connect to the database"""
        await self.sql_server.connect()
        
    async def cleanup(self):
        """Clean up resources"""
        await self.sql_server.close()

async def main():
    server = MCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 