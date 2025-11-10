import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()  # load environment variables from .env

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure loguru
logger.add(
    "logs/mcp_client_{time}.log", rotation="500 MB", retention="10 days", level="DEBUG"
)
logger.info("MCP Client module initialized")


@dataclass
class ServerConfig:
    """Configuration for a single MCP server"""

    name: str
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None
    type: Optional[str] = None  # "sse" or "streamable-http"
    url: Optional[str] = None
    headers: Optional[dict[str, str]] = None
    alwaysAllow: Optional[list[str]] = None
    disabled: bool = False
    description: Optional[str] = None


class MCPServerManager:
    """Manages multiple MCP server connections"""

    def __init__(self, exit_stack: AsyncExitStack):
        self.exit_stack = exit_stack
        self.sessions: dict[str, ClientSession] = {}
        self.server_configs: dict[str, ServerConfig] = {}
        logger.debug("MCPServerManager initialized")

    @staticmethod
    def load_config(config_path: str) -> dict[str, ServerConfig]:
        """Load server configurations from JSON file

        Args:
            config_path: Path to the configuration JSON file

        Returns:
            Dictionary mapping server names to ServerConfig objects

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        logger.info(f"Loading MCP server configuration from: {config_path}")

        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            if "mcpServers" not in config_data:
                logger.error("Invalid config: missing 'mcpServers' key")
                raise ValueError("Invalid config file: missing 'mcpServers' key")

            servers = {}
            for name, server_data in config_data["mcpServers"].items():
                if server_data.get("disabled", False):
                    logger.info(f"Skipping disabled server: {name}")
                    continue

                servers[name] = ServerConfig(
                    name=name,
                    command=server_data.get("command"),
                    args=server_data.get("args"),
                    env=server_data.get("env"),
                    type=server_data.get("type"),
                    url=server_data.get("url"),
                    headers=server_data.get("headers"),
                    alwaysAllow=server_data.get("alwaysAllow"),
                    disabled=server_data.get("disabled", False),
                    description=server_data.get("description"),
                )
                logger.debug(f"Loaded config for server: {name}")

            logger.info(f"Successfully loaded {len(servers)} server configuration(s)")
            return servers

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.exception(f"Error loading config file: {e}")
            raise

    async def connect_server(self, config: ServerConfig) -> ClientSession:
        """Connect to a single MCP server

        Args:
            config: Server configuration

        Returns:
            Connected ClientSession

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info(f"Connecting to server: {config.name}")

        try:
            # Stdio transport (command-based)
            if config.command:
                logger.debug(
                    f"Using stdio transport - Command: {config.command} {config.args}"
                )

                # Merge environment variables
                env = os.environ.copy()
                if config.env:
                    env.update(config.env)
                    logger.debug(f"Added {len(config.env)} environment variable(s)")

                server_params = StdioServerParameters(
                    command=config.command, args=config.args or [], env=env
                )

                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                logger.info(f"Stdio transport established for: {config.name}")

            # SSE transport
            elif config.type == "sse" and config.url:
                logger.debug(f"Using SSE transport - URL: {config.url}")

                sse_transport = await self.exit_stack.enter_async_context(
                    sse_client(config.url)
                )
                stdio, write = sse_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                logger.info(f"SSE transport established for: {config.name}")

            # Streamable HTTP (if supported by MCP SDK)
            elif config.type == "streamable-http" and config.url:
                logger.warning(
                    f"streamable-http transport not yet implemented for: {config.name}"
                )
                raise ValueError(
                    f"Transport type 'streamable-http' is not yet supported for server: {config.name}"
                )

            else:
                logger.error(f"Invalid server configuration for: {config.name}")
                raise ValueError(
                    f"Invalid server configuration for {config.name}: "
                    f"must provide either 'command' or 'type' with 'url'"
                )

            # Initialize the session
            logger.debug(f"Initializing session for: {config.name}")
            await session.initialize()
            logger.info(f"Session initialized successfully for: {config.name}")

            return session

        except Exception as e:
            logger.exception(f"Failed to connect to server {config.name}: {e}")
            raise

    async def connect_all(self, configs: dict[str, ServerConfig]):
        """Connect to all configured servers

        Args:
            configs: Dictionary of server configurations
        """
        logger.info(f"Connecting to {len(configs)} server(s)")

        self.server_configs = configs

        # Connect to all servers in parallel
        connection_tasks = []
        for name, config in configs.items():
            connection_tasks.append(self._connect_and_store(name, config))

        results = await asyncio.gather(*connection_tasks, return_exceptions=True)

        # Log connection results
        success_count = 0
        for name, result in zip(configs.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to {name}: {result}")
            else:
                success_count += 1

        logger.info(
            f"Connected to {success_count}/{len(configs)} server(s) successfully"
        )

        if success_count == 0:
            raise RuntimeError("Failed to connect to any MCP servers")

    async def _connect_and_store(self, name: str, config: ServerConfig):
        """Helper method to connect and store a session"""
        try:
            session = await self.connect_server(config)
            self.sessions[name] = session
            logger.debug(f"Stored session for server: {name}")
        except Exception as e:
            logger.error(f"Could not connect to server {name}: {e}")
            raise

    async def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools from all connected servers with prefixed names

        Returns:
            List of tools in OpenAI format with server-prefixed names
        """
        logger.debug("Fetching tools from all servers")

        all_tools = []
        for server_name, session in self.sessions.items():
            try:
                response = await session.list_tools()
                tools = response.tools
                logger.debug(f"Server '{server_name}' has {len(tools)} tool(s)")

                for tool in tools:
                    # Add server prefix to tool name
                    prefixed_name = f"{server_name}__{tool.name}"

                    # Convert to OpenAI format with prefixed name
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": prefixed_name,
                            "description": f"[{server_name}] {tool.description}",
                            "parameters": tool.inputSchema,
                        },
                    }
                    all_tools.append(openai_tool)
                    logger.debug(f"Added tool: {prefixed_name}")

            except Exception as e:
                logger.error(f"Failed to get tools from server {server_name}: {e}")
                # Continue with other servers even if one fails

        logger.info(f"Total tools available: {len(all_tools)}")
        return all_tools

    async def call_tool(
        self, prefixed_tool_name: str, arguments: dict
    ) -> Any:
        """Route tool call to the appropriate server

        Args:
            prefixed_tool_name: Tool name with server prefix (e.g., "crawl4ai__crawl_url")
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool name format is invalid or server not found
        """
        logger.debug(f"Routing tool call: {prefixed_tool_name}")

        # Parse server name and tool name
        if "__" not in prefixed_tool_name:
            logger.error(f"Invalid tool name format: {prefixed_tool_name}")
            raise ValueError(
                f"Invalid tool name format: {prefixed_tool_name}. "
                f"Expected format: server_name__tool_name"
            )

        server_name, tool_name = prefixed_tool_name.split("__", 1)
        logger.debug(f"Parsed - Server: {server_name}, Tool: {tool_name}")

        if server_name not in self.sessions:
            logger.error(f"Server not found: {server_name}")
            raise ValueError(f"Server '{server_name}' not found or not connected")

        session = self.sessions[server_name]
        logger.info(f"Calling tool '{tool_name}' on server '{server_name}'")

        try:
            result = await session.call_tool(tool_name, arguments)
            logger.debug(f"Tool call successful: {tool_name}")
            return result
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name} on {server_name}: {e}")
            raise


class MCPClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, max_tokens: int = 4000, max_iterations: int = 10):
        # Initialize session and client objects
        logger.debug("Initializing MCPClient")
        self.exit_stack = AsyncExitStack()
        self.server_manager = MCPServerManager(self.exit_stack)

        # Try to get from parameters first, then fall back to environment variables
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model or os.getenv("MODEL")
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4000"))
        self.max_iterations = max_iterations or int(os.getenv("MAX_ITERATIONS", "10"))

        # OpenAI client will be created when needed (lazy initialization)
        self.openai = None

        logger.info("MCPClient initialized (API config will be set when needed)")

    def set_api_config(self, api_key: str, base_url: str, model: str):
        """
        Set or update API configuration

        Args:
            api_key: OpenAI API key
            base_url: OpenAI base URL
            model: Model name
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # Create/recreate OpenAI client
        self.openai = OpenAI(api_key=self.api_key, base_url=self.base_url)

        logger.info(f"API config updated - Base URL: {self.base_url}, Model: {self.model}")

    def _ensure_openai_client(self):
        """Ensure OpenAI client is initialized"""
        if self.openai is None:
            if not all([self.api_key, self.base_url, self.model]):
                raise ValueError(
                    "API configuration not set. Please call set_api_config() first or provide "
                    "API_KEY, BASE_URL, and MODEL environment variables."
                )
            self.openai = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.debug(f"OpenAI client created - Base URL: {self.base_url}, Model: {self.model}")

    async def connect_to_servers(self, config_path: str):
        """Connect to multiple MCP servers from configuration file

        Args:
            config_path: Path to the JSON configuration file
        """
        logger.info(f"Connecting to servers from config: {config_path}")

        configs = MCPServerManager.load_config(config_path)
        await self.server_manager.connect_all(configs)

        # Log connected servers
        server_names = list(self.server_manager.sessions.keys())
        logger.info(f"Connected servers: {server_names}")

    async def process_query(self, query: str, messages: Optional[list] = None, callback=None) -> tuple[str, list]:
        """Process a query using OpenAI and available tools

        Args:
            query: User query string (required)
            messages: Optional list of previous messages for conversation history
            callback: Optional callback function(event_type, data) for real-time updates
                event_type can be: "thinking", "tool_call_start", "tool_call_end", "response", "error"

        Returns:
            Tuple of (final_response, updated_messages)
        """
        logger.info(
            f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}"
        )

        # Ensure OpenAI client is initialized
        self._ensure_openai_client()

        # Initialize or use provided messages
        if messages is None:
            messages = []

        # Append the new user query
        messages.append({"role": "user", "content": query})

        logger.debug("Fetching available tools from all servers")

        # Get all tools from server manager
        available_tools = await self.server_manager.get_all_tools()

        if not available_tools:
            logger.error("No tools available from connected servers")
            raise RuntimeError(
                "No tools available. Ensure servers are connected via connect_to_servers() "
                "and at least one server provides tools."
            )

        logger.info(
            f"Using {len(self.server_manager.sessions)} server(s) "
            f"with {len(available_tools)} tool(s) total"
        )

        # Notify thinking started
        if callback:
            await callback("thinking", {"status": "AI is thinking..."})

        # Initial OpenAI API call
        logger.debug("Calling OpenAI API for initial response")
        response = self.openai.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=available_tools,
        )
        logger.debug("Received initial response from OpenAI")

        # Process response and handle tool calls
        final_text = []
        iteration = 0

        try:
            while iteration < self.max_iterations:
                iteration += 1
                logger.debug(f"Processing iteration {iteration}/{self.max_iterations}")
                message = response.choices[0].message

                # If no tool calls, we're done
                if not message.tool_calls:
                    logger.debug("No tool calls in response, completing query")
                    if message.content:
                        final_text.append(message.content)
                        if callback:
                            await callback("response", {"content": message.content})
                    break

                logger.info(f"Processing {len(message.tool_calls)} tool call(s)")

                # Add assistant message to conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name

                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Executing tool: {tool_name}")
                    logger.debug(f"Tool arguments: {tool_args}")

                    # Notify tool call started
                    if callback:
                        await callback(
                            "tool_call_start",
                            {"tool_name": tool_name, "arguments": tool_args},
                        )

                    start_time = time.time()
                    try:
                        # Route tool call to appropriate server
                        result = await self.server_manager.call_tool(tool_name, tool_args)
                        elapsed_time = time.time() - start_time
                        logger.info(
                            f"Tool {tool_name} completed in {elapsed_time:.2f}s"
                        )
                        result_str = str(result.content)
                        logger.debug(f"Tool result preview: {result_str[:200]}...")

                        # Notify tool call completed
                        if callback:
                            await callback(
                                "tool_call_end",
                                {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "result": result_str,
                                    "elapsed_time": elapsed_time,
                                },
                            )

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_str,
                            }
                        )
                    except Exception as e:
                        error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
                        logger.error(error_msg)

                        # Notify callback of error
                        if callback:
                            await callback(
                                "error",
                                {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "error": str(e),
                                },
                            )
                        raise

                # Notify thinking again
                if callback:
                    await callback("thinking", {"status": "Processing results..."})

                # Get next response from OpenAI
                logger.debug("Calling OpenAI API for next response")
                response = self.openai.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    tools=available_tools,
                )
                logger.debug("Received next response from OpenAI")

            # Check if we hit max iterations
            if iteration >= self.max_iterations:
                warning_msg = f"Reached maximum iterations ({self.max_iterations}). Processing may be incomplete."
                logger.warning(warning_msg)
                if callback:
                    await callback("error", {"error": warning_msg})

            final_result = "\n".join(final_text)
            logger.info(f"Query processing completed after {iteration} iteration(s)")
            logger.debug(f"Final result length: {len(final_result)} characters")
            return final_result, messages

        except Exception as e:
            logger.exception(f"Error during query processing: {str(e)}")
            raise

    async def chat_loop(self):
        """Run an interactive chat loop with conversation history"""
        logger.info("Starting interactive chat loop")

        query_count = 0
        messages = []  # Maintain conversation history across queries

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    logger.info("User requested to quit chat loop")
                    break

                query_count += 1
                logger.info(f"Query #{query_count} received")
                response, messages = await self.process_query(query, messages=messages)
                # Use print here as this is the actual output to user in interactive mode
                print("\n" + response)

            except KeyboardInterrupt:
                logger.info("Chat loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                # Use print here to show error to user in interactive mode
                print(f"\nError: {str(e)}")

        logger.info(f"Chat loop ended. Total queries processed: {query_count}")

    async def process_query_streaming(
        self, query: str, messages: Optional[list] = None
    ):
        """
        流式处理查询，实时生成事件

        Args:
            query: 用户查询
            messages: 可选的对话历史

        Yields:
            事件字典，包含以下类型：
            - {"type": "content", "data": str} - LLM 生成的文本内容
            - {"type": "tool_call_start", "tool_name": str, "arguments": dict} - 工具调用开始
            - {"type": "tool_call_end", "tool_name": str, "result": str} - 工具调用完成
            - {"type": "thinking", "status": str} - AI 思考状态
            - {"type": "error", "error": str} - 错误信息
            - {"type": "done"} - 处理完成
        """
        logger.info(
            f"Processing streaming query: {query[:100]}{'...' if len(query) > 100 else ''}"
        )

        # Ensure OpenAI client is initialized
        try:
            self._ensure_openai_client()
        except ValueError as e:
            logger.error(f"API configuration error: {e}")
            yield {"type": "error", "error": str(e)}
            return

        # Initialize or use provided messages
        if messages is None:
            messages = []

        # Append the new user query
        messages.append({"role": "user", "content": query})

        logger.debug("Fetching available tools from all servers")

        # Get all tools from server manager
        available_tools = await self.server_manager.get_all_tools()

        if not available_tools:
            logger.error("No tools available from connected servers")
            yield {
                "type": "error",
                "error": "No tools available. Ensure servers are connected.",
            }
            return

        logger.info(
            f"Using {len(self.server_manager.sessions)} server(s) "
            f"with {len(available_tools)} tool(s) total"
        )

        iteration = 0

        try:
            while iteration < self.max_iterations:
                iteration += 1
                logger.debug(f"Processing iteration {iteration}/{self.max_iterations}")

                # Notify thinking
                yield {"type": "thinking", "status": "AI is thinking..."}

                # Call OpenAI API with streaming
                logger.debug("Calling OpenAI API for streaming response")
                response = self.openai.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    tools=available_tools,
                    stream=True,  # Enable streaming
                )

                # Collect complete message including tool_calls
                full_message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": None,
                }
                tool_calls_dict = {}  # Dictionary to accumulate tool call fragments

                # Process streaming response
                for chunk in response:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # Stream text content
                    if delta.content:
                        full_message["content"] += delta.content
                        yield {"type": "content", "data": delta.content}

                    # Accumulate tool calls (OpenAI sends tool_calls in fragments)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            if tc_delta.id:
                                tool_calls_dict[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_dict[idx]["function"][
                                        "name"
                                    ] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_dict[idx]["function"][
                                        "arguments"
                                    ] += tc_delta.function.arguments

                # Convert tool_calls_dict to list if present
                if tool_calls_dict:
                    full_message["tool_calls"] = [
                        tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())
                    ]

                # If no tool calls, we're done
                if not full_message["tool_calls"]:
                    logger.debug("No tool calls in response, completing query")
                    yield {"type": "done"}
                    return

                logger.info(f"Processing {len(full_message['tool_calls'])} tool call(s)")

                # Add assistant message to conversation
                messages.append(full_message)

                # Execute each tool call
                for tool_call in full_message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"Executing tool: {tool_name}")
                    logger.debug(f"Tool arguments: {tool_args}")

                    # Notify tool call started
                    yield {
                        "type": "tool_call_start",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                    }

                    start_time = time.time()
                    try:
                        # Route tool call to appropriate server
                        result = await self.server_manager.call_tool(
                            tool_name, tool_args
                        )
                        elapsed_time = time.time() - start_time
                        logger.info(f"Tool {tool_name} completed in {elapsed_time:.2f}s")
                        result_str = str(result.content)
                        logger.debug(f"Tool result preview: {result_str[:200]}...")

                        # Notify tool call completed
                        yield {
                            "type": "tool_call_end",
                            "tool_name": tool_name,
                            "result": result_str[:500],  # Truncate for display
                            "elapsed_time": elapsed_time,
                        }

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": result_str,
                            }
                        )
                    except Exception as e:
                        error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
                        logger.error(error_msg)
                        yield {"type": "error", "error": error_msg}
                        raise

            # Check if we hit max iterations
            if iteration >= self.max_iterations:
                warning_msg = f"Reached maximum iterations ({self.max_iterations}). Processing may be incomplete."
                logger.warning(warning_msg)
                yield {"type": "error", "error": warning_msg}

            yield {"type": "done"}

        except Exception as e:
            logger.exception(f"Error during streaming query processing: {str(e)}")
            yield {"type": "error", "error": str(e)}

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Starting cleanup process")
        try:
            await self.exit_stack.aclose()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise


async def main():
    logger.info("MCP Client application starting")
    logger.debug(f"Command line arguments: {sys.argv}")

    # Default config file path
    default_config = "configs/mcp_config.json"

    # Determine config file path
    if len(sys.argv) < 2:
        # No arguments - try to use default config file
        if Path(default_config).exists():
            config_path = default_config
            logger.info(f"No config file specified, using default: {default_config}")
        else:
            logger.warning("No config file specified and default not found")
            print("Usage: python mcp_client.py [config_file.json]")
            print("\nExamples:")
            print("  python mcp_client.py                    # Uses mcp_config.json")
            print("  python mcp_client.py my_config.json     # Uses specified config")
            print("\nConfig file format: See MULTI_SERVER_GUIDE.md for details")
            sys.exit(1)
    else:
        config_path = sys.argv[1]

    logger.info(f"Loading configuration from: {config_path}")

    client = MCPClient()
    try:
        await client.connect_to_servers(config_path)
        await client.chat_loop()
    except Exception as e:
        logger.exception(f"Fatal error in main: {str(e)}")
        raise
    finally:
        logger.info("Shutting down application")
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
