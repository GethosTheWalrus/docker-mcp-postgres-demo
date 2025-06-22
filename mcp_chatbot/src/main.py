import os
import asyncio
import time
import traceback
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

from chat import get_chat_model
from mcp_tools import initialize_mcp, get_available_tools, execute_mcp_tool, cleanup_mcp

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, List

# ANSI color codes
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def print_greeting():
    """Print the greeting message in a box."""
    print("╭──────────────────────────────────────────────────────────────────────────────╮")
    print("│                            MCP Agent Chatbot                                │")
    print("╰────────────── Powered by LangChain Agents, Docker and Ollama ──────────────╯")

def print_thinking():
    """Print a loading indicator with spinning animation."""
    print(f"{CYAN}Agent >{RESET} Thinking", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" ")

def print_help():
    """Print available commands."""
    print("\nAvailable commands:")
    print("  /tools          - List available MCP tools")
    print("  /memory         - Show conversation memory")
    print("  /new            - Start a new conversation (clear memory)")
    print("  /help           - Show this help message")
    print("  exit            - Exit the chatbot")

class MCPToolWrapper:
    """Wrapper to convert MCP tools to LangChain tools"""
    
    def __init__(self, mcp_manager):
        self.mcp_manager = mcp_manager
        self.tools_cache = None
    
    async def get_langchain_tools(self):
        """Convert MCP tools to LangChain tools"""
        if self.tools_cache is not None:
            return self.tools_cache
            
        mcp_tools = await self.mcp_manager.client.get_tools()
        langchain_tools = []
        
        for mcp_tool in mcp_tools:
            # Create a LangChain tool for each MCP tool
            langchain_tool = self._create_langchain_tool(mcp_tool)
            langchain_tools.append(langchain_tool)
        
        self.tools_cache = langchain_tools
        return langchain_tools
    
    def _create_langchain_tool(self, mcp_tool):
        """Create a LangChain tool from an MCP tool"""
        
        # Get the server name for the tool (if available)
        server_name = getattr(mcp_tool, 'server', None)
        
        # If server name is not available, and we only have one server, use that
        if not server_name and len(self.mcp_manager.config["mcpServers"]) == 1:
            server_name = list(self.mcp_manager.config["mcpServers"].keys())[0]
            print(f"DEBUG - Using single server: '{server_name}'")
        
        # Create a unique tool name
        tool_name = f"{server_name}_{mcp_tool.name}" if server_name else mcp_tool.name
        
        # Use the exact description from the MCP tool, with optional server description
        description = mcp_tool.description
        
        # Add server description if available in config
        if server_name and server_name in self.mcp_manager.config["mcpServers"]:
            server_config = self.mcp_manager.config["mcpServers"][server_name]
            if "description" in server_config:
                description += f" ({server_config['description']})"
                print(f"DEBUG - Enhanced description: {description}")
        
        # Get the input schema from the MCP tool
        input_schema = getattr(mcp_tool, 'args_schema', None)
        
        # Debug: Print the actual schema
        print(f"DEBUG - Tool: {tool_name}")
        print(f"DEBUG - Description: {description}")
        print(f"DEBUG - Input Schema: {input_schema}")
        print(f"DEBUG - Schema Type: {type(input_schema)}")
        print("---")
        
        async def dynamic_tool_func(**kwargs) -> str:
            """Dynamically created function that calls the MCP tool"""
            error_result = None
            
            try:
                print(f"DEBUG - Tool called with kwargs: {kwargs}")
                
                # Use the MCP tool directly via ainvoke (the langchain-mcp-adapters way)
                result = await mcp_tool.ainvoke(kwargs)
                print(f"DEBUG - MCP tool returned: {result}")
                
                # Convert result to string for LangChain agent
                if isinstance(result, dict):
                    return json.dumps(result, indent=2)
                elif isinstance(result, list):
                    return json.dumps(result, indent=2)
                else:
                    return str(result)
                    
            except* Exception as e:
                print(f"DEBUG - MCP tool error: {e}")
                print(f"DEBUG - Exception type: {type(e)}")
                
                # Handle both single exceptions and exception groups
                real_errors = []
                if hasattr(e, 'exceptions') and e.exceptions:
                    print(f"DEBUG - ExceptionGroup with {len(e.exceptions)} exceptions")
                    for i, exc in enumerate(e.exceptions):
                        print(f"DEBUG - Exception {i}: {type(exc).__name__}: {exc}")
                        traceback.print_exception(type(exc), exc, exc.__traceback__)
                        real_errors.append({
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)
                        })
                else:
                    # Single exception wrapped in except*
                    print(f"DEBUG - Single exception: {type(e).__name__}: {e}")
                    traceback.print_exception(type(e), e, e.__traceback__)
                    real_errors.append({
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exception(type(e), e, e.__traceback__)
                    })
                
                # Build comprehensive error info for the agent
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "tool_name": tool_name,
                    "attempted_args": kwargs,
                    "expected_schema": input_schema,
                    "underlying_errors": real_errors,
                    "exception_count": len(real_errors)
                }
                
                # Prioritize the first real error for the agent with actionable message
                if real_errors:
                    primary_error = real_errors[0]
                    
                    # Extract clean error message
                    error_msg = primary_error['message']
                    if ':' in error_msg and any(prefix in error_msg for prefix in ['McpError:', 'Error:', 'Exception:']):
                        # Extract just the core error message after the error type
                        clean_error = error_msg.split(':')[-1].strip()
                        error_result = f"TOOL_ERROR: {clean_error}\n\nPlease analyze this error and try a different approach with corrected parameters."
                    else:
                        error_result = f"TOOL_ERROR: {error_msg}\n\nPlease analyze this error and try a different approach."
                else:
                    error_result = f"TOOL_ERROR: {json.dumps(error_info, indent=2)}"
            
            # Return error if we caught one
            if error_result:
                return error_result
        
        # Use the MCP schema directly if available
        if input_schema and isinstance(input_schema, dict):
            # Convert JSON Schema to Pydantic dynamically
            from pydantic import create_model
            
            # Extract field definitions from JSON Schema
            fields = {}
            if 'properties' in input_schema:
                for field_name, field_def in input_schema['properties'].items():
                    field_type = str  # Default to string, could be enhanced to handle other types
                    if field_def.get('type') == 'string':
                        field_type = str
                    elif field_def.get('type') == 'integer':
                        field_type = int
                    elif field_def.get('type') == 'boolean':
                        field_type = bool
                    
                    # Add field with description if available
                    fields[field_name] = (field_type, Field(description=field_def.get('description', f'{field_name} parameter')))
            
            # Create dynamic Pydantic model
            DynamicInputModel = create_model(f'{tool_name}Input', **fields)
            
            # Create tool with dynamic schema
            langchain_tool = StructuredTool.from_function(
                func=dynamic_tool_func,
                name=tool_name,
                description=description,
                args_schema=DynamicInputModel,
                coroutine=dynamic_tool_func
            )
        else:
            # No schema available, create basic tool
            langchain_tool = StructuredTool.from_function(
                func=dynamic_tool_func,
                name=tool_name,
                description=description,
                coroutine=dynamic_tool_func
            )
        
        return langchain_tool

async def main():
    """
    Main function for the MCP Agent Chatbot.
    """
    load_dotenv()
    print_greeting()
    
    # Initialize components
    try:
        print("🔧 Initializing chat model...")
        chat_model = get_chat_model()
        
        print("🔧 Initializing MCP servers...")
        mcp_manager = await initialize_mcp()
        print("✅ MCP initialization complete!")
        
        # Wrap MCP tools for LangChain
        print("🔧 Setting up agent tools...")
        tool_wrapper = MCPToolWrapper(mcp_manager)
        tools = await tool_wrapper.get_langchain_tools()
        print(f"📋 Found {len(tools)} MCP tools")
        
        # Print available tools
        print("\n🛠️  Available tools:")
        for tool in tools:
            print(f"  - {GREEN}{tool.name}{RESET}: {tool.description}")
        
        # Set up memory for conversation history
        memory = MemorySaver()
        
        # Create the ReAct agent with enhanced system prompt
        print("🤖 Creating ReAct agent...")
        
        # Enhanced system prompt function for better reasoning and persistence
        def create_system_prompt(state):
            """Create a dynamic system prompt for the agent"""
            system_content = """You are an intelligent assistant that can use various tools to help users. 

Key principles:
1. ALWAYS use tool results to answer questions - never make assumptions or provide generic responses
2. When you get JSON results from tools, parse them and present the information in a user-friendly format
3. When tools fail with errors, IMMEDIATELY try again with corrected parameters or approach
4. Analyze error messages carefully to understand what went wrong and how to fix it
5. Be VERY persistent - keep trying different approaches until you succeed
6. Never give up after one failed attempt - always try at least 2-3 different approaches
7. Always read tool descriptions carefully to understand what parameters they expect and what system you're working with
8. Chain multiple tool calls together when needed to fully answer complex questions
9. Think step by step and reason through problems systematically

CRITICAL RULES:
- After using a tool successfully, you MUST interpret and present the results to the user
- When you see a TOOL_ERROR, immediately retry with corrected parameters based on the error message
- Pay attention to the tool descriptions for context about what system/database/API you're working with
- For complex tasks, break them down into multiple tool calls and chain them together
- Keep trying until you get a successful result or exhaust reasonable options
- Always provide a complete, helpful response based on the tool results"""
            
            from langchain_core.messages import SystemMessage
            return [SystemMessage(content=system_content)] + state["messages"]

        agent_executor = create_react_agent(
            model=chat_model,
            tools=tools,
            checkpointer=memory,
            prompt=create_system_prompt
        )
        
        print("✅ Agent ready! The agent can now reason through problems and iterate when tools fail.")
        
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return

    print_help()
    
    # Conversation thread ID for memory
    thread_id = "default_conversation"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input(f"\n{CYAN}You >{RESET} ")
        
        if user_input.lower() == 'exit':
            break

        if user_input.startswith("/help"):
            print_help()
            continue
            
        if user_input.startswith("/tools"):
            print("\n📋 Available MCP Tools:")
            for tool in tools:
                print(f"  - {GREEN}{tool.name}{RESET}: {tool.description}")
            continue
            
        if user_input.startswith("/memory"):
            print(f"\n🧠 Current conversation thread: {thread_id}")
            print("Memory is preserved across messages in this session.")
            continue
            
        if user_input.startswith("/new"):
            # Generate new thread ID to start fresh conversation
            import uuid
            thread_id = str(uuid.uuid4())[:8]
            config = {"configurable": {"thread_id": thread_id}}
            print(f"🆕 Started new conversation (thread: {thread_id})")
            continue

        # Handle regular chat with agent
        print_thinking()
        
        try:
            # Use the agent with proper ReAct flow - let it run until completion
            print(f"\n{GREEN}Agent >{RESET} ")
            
            # The agent should handle the full conversation flow
            result = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            )
            
            # Extract and display the final response
            if "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content') and final_message.content:
                    print(final_message.content)
                else:
                    print("Agent completed but provided no final response.")
            else:
                print("No response from agent.")
                
        except Exception as e:
            print(f"\n{RED}❌ Error during conversation: {e}{RESET}")
            print("The agent encountered an error. Please try rephrasing your question.")

    # Cleanup
    print(f"\n{CYAN}🧹 Cleaning up...{RESET}")
    await cleanup_mcp()
    print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main()) 