"""
Crawl4AI MCP Server

A Model Context Protocol (MCP) server that provides web crawling capabilities
using crawl4ai. Allows AI assistants to fetch and extract web page content.
"""
import asyncio
import os
import sys
from typing import Optional

import crawl4ai
from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

os.environ['DEBUG'] = ''

# Redirect stdout to stderr to prevent crawl4ai output from polluting MCP messages
# MCP stdio mode requires stdout to be used exclusively for JSON-RPC messages
original_stdout = sys.stdout
sys.stdout = sys.stderr

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure loguru
logger.add(
    "logs/crawl4ai_mcp_server_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG",
)
logger.info("Crawl4AI MCP Server module initialized")

# Create MCP server
server = Server("crawl4ai-server")

# Global crawler instance - initialized once and reused for all requests
crawler: Optional[crawl4ai.AsyncWebCrawler] = None


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available MCP tools.

    Returns:
        List of available tools with their schemas
    """
    logger.debug("Listing available tools")
    return [
        types.Tool(
            name="crawl_url",
            description="Crawl a web page and return its content in markdown format. "
            "This tool fetches the specified URL and extracts the main content, "
            "converting it to markdown format suitable for LLM consumption.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to crawl (must include http:// or https://)",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["url"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.

    Args:
        name: Name of the tool to execute
        arguments: Tool arguments

    Returns:
        List of content items (text, image, or embedded resources)

    Raises:
        ValueError: If tool name is unknown or arguments are invalid
    """
    logger.info(f"Tool called: {name}")
    logger.debug(f"Arguments: {arguments}")

    if name == "crawl_url":
        url = arguments.get("url")
        timeout = arguments.get("timeout", 30)

        if not url:
            logger.error("URL parameter is required")
            raise ValueError("URL parameter is required")

        logger.info(f"Crawling URL: {url} (timeout: {timeout}s)")

        try:
            # Use the global crawler instance
            result = await crawler.arun(url=url, config=crawl4ai.CrawlerRunConfig(verbose=False))

            # Check if the crawl was successful
            if not result.success:
                error_msg = f"Failed to access {url}. The page may be unavailable or blocking automated access."
                logger.error(f"{error_msg} Error details: {result.error_message if hasattr(result, 'error_message') else 'Unknown'}")
                raise ValueError(error_msg)

            markdown_content = result.markdown

            # Check if markdown content is None or empty
            if markdown_content is None or len(markdown_content.strip()) == 0:
                logger.warning(f"No markdown content extracted from {url}")
                # Try to fall back to HTML content
                if result.html and len(result.html.strip()) > 0:
                    logger.info(f"Falling back to HTML content for {url}")
                    markdown_content = result.html
                else:
                    error_msg = f"Failed to extract any content from {url}. The page may be empty or require JavaScript rendering."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            logger.info(
                f"Successfully crawled {url}, content length: {len(markdown_content)} characters"
            )
            logger.debug(f"Content preview: {markdown_content[:200]}...")

            return [
                types.TextContent(
                    type="text",
                    text=markdown_content,
                )
            ]

        except ValueError:
            # Re-raise ValueError (our custom errors)
            raise
        except Exception as e:
            # Catch all other exceptions
            error_msg = f"Unexpected error while crawling {url}: {str(e)}"
            logger.error(error_msg)
            logger.exception("Crawling exception details:")
            raise ValueError(error_msg)

    else:
        error_msg = f"Unknown tool: {name}"
        logger.error(error_msg)
        raise ValueError(error_msg)


async def main():
    """
    Main function to initialize and run the MCP server.

    This function:
    1. Initializes the global AsyncWebCrawler instance (starts browser once)
    2. Runs the stdio MCP server
    3. Cleans up the crawler on shutdown
    """
    global crawler

    logger.info("Starting Crawl4AI MCP Server")

    try:
        # Initialize crawler once - this starts the browser
        logger.info("Initializing AsyncWebCrawler...")
        crawler = crawl4ai.AsyncWebCrawler()
        await crawler.__aenter__()
        logger.info("AsyncWebCrawler initialized successfully (browser started)")

        # Restore stdout for MCP communication
        sys.stdout = original_stdout
        logger.info("Restored stdout for MCP communication")

        # Run the stdio server
        logger.info("Starting stdio server...")
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Stdio server running, waiting for requests...")
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    except Exception as e:
        logger.exception(f"Fatal error in main: {str(e)}")
        raise

    finally:
        # Clean up crawler - this closes the browser
        logger.info("Shutting down server...")
        if crawler:
            logger.info("Closing AsyncWebCrawler (shutting down browser)...")
            try:
                await crawler.__aexit__(None, None, None)
                logger.info("AsyncWebCrawler closed successfully")
            except Exception as e:
                logger.error(f"Error closing crawler: {str(e)}")

        logger.info("Crawl4AI MCP Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
