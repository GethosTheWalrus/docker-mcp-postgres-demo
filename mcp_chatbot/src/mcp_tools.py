import json
import asyncio
from typing import Dict, List, Any
from langchain_mcp_adapters.client import MultiServerMCPClient

class MCPManager:
    def __init__(self, config_path="mcp-config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.client = None

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        # Add transport if not present (default to stdio)
        for server in config.get("mcpServers", {}).values():
            if "transport" not in server:
                server["transport"] = "stdio"
        return config

    async def start(self):
        # The MultiServerMCPClient expects a dict of servers
        self.client = MultiServerMCPClient(self.config["mcpServers"])
        # No need to call start() - the client is ready to use

    async def get_tools(self) -> Dict[str, List[str]]:
        tools = await self.client.get_tools()
        # Group tools by server
        grouped = {}
        for tool in tools:
            # Check if tool has server attribute, otherwise use a default
            server_name = getattr(tool, 'server', 'unknown')
            grouped.setdefault(server_name, []).append(tool.name)
        return grouped

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return await self.client.call_tool(server_name, tool_name, arguments)

    async def close(self):
        if self.client:
            await self.client.aclose()

# Singleton instance
mcp_manager: MCPManager = None

async def initialize_mcp():
    global mcp_manager
    mcp_manager = MCPManager()
    await mcp_manager.start()
    return mcp_manager

async def get_available_tools() -> Dict[str, List[str]]:
    if not mcp_manager:
        await initialize_mcp()
    return await mcp_manager.get_tools()

async def execute_mcp_tool(server_name: str, tool_name: str, arguments: Dict) -> Any:
    if not mcp_manager:
        await initialize_mcp()
    return await mcp_manager.call_tool(server_name, tool_name, arguments)

async def cleanup_mcp():
    if mcp_manager:
        await mcp_manager.close() 