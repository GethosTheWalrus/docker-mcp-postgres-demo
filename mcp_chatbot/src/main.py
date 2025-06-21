import os
import asyncio
import time
from dotenv import load_dotenv

from chat import get_chat_model
from mcp_tools import initialize_mcp, get_available_tools, execute_mcp_tool, cleanup_mcp

from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# ANSI color codes
CYAN = '\033[96m'
RESET = '\033[0m'

def print_greeting():
    """Print the greeting message in a box."""
    print("╭──────────────────────────────────────────────────────────────────────────────╮")
    print("│                                 MCP Chatbot                                  │")
    print("╰────────────────────────── Powered by Docker and Ollama ──────────────────────╯")

def print_thinking():
    """Print a loading indicator with spinning animation."""
    print(f"{CYAN}AI >{RESET} Thinking", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" ", end="", flush=True)

def print_help():
    """Print available commands."""
    print("\nAvailable commands:")
    print("  /tools          - List available MCP tools")
    print("  /mcp <server>   - List tools for a specific server")
    print("  /call <server> <tool> [args] - Call a specific tool")
    print("  /help           - Show this help message")
    print("  exit            - Exit the chatbot")

async def main():
    """
    Main function for the MCP Chatbot.
    """
    load_dotenv()
    print_greeting()
    
    try:
        chat_model = get_chat_model()
        print("🔧 Initializing MCP servers...")
        mcp_manager = await initialize_mcp()
        print("✅ MCP initialization complete!")
        # Get all MCP tools
        tools = await mcp_manager.client.get_tools()
        print(f"📋 Found {len(tools)} MCP tools")
        
        # Debug: Print tool details
        for tool in tools:
            print(f"🔧 Tool: {tool.name}")
            print(f"   Description: {tool.description}")
            print(f"   Args Schema: {getattr(tool, 'args_schema', 'No schema')}")
            print(f"   Tool object: {type(tool)}")
            print("---")
        
        # Instead of using LangChain agents, let's create a simple custom implementation
        # that directly handles tool calls to avoid JSON conversion issues
        
        print("🤖 Simple MCP chatbot ready! Type your question.")
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return

    print_help()

    while True:
        user_input = input(f"{CYAN}You >{RESET} ")
        if user_input.lower() == 'exit':
            break

        if user_input.startswith("/help"):
            print_help()
            continue
            
        if user_input.startswith("/tools"):
            try:
                tools_dict = await get_available_tools()
                if tools_dict:
                    print("\n📋 Available MCP Tools:")
                    for server, tool_list in tools_dict.items():
                        print(f"\n  {server}:")
                        for tool in tool_list:
                            print(f"    - {tool}")
                else:
                    print("No MCP tools available.")
            except Exception as e:
                print(f"Error listing tools: {e}")
            continue

        if user_input.startswith("/mcp"):
            parts = user_input.split()
            if len(parts) == 2:
                server_name = parts[1]
                try:
                    tools_dict = await get_available_tools()
                    if server_name in tools_dict:
                        print(f"\n📋 Tools for {server_name}:")
                        for tool in tools_dict[server_name]:
                            print(f"  - {tool}")
                    else:
                        print(f"Server '{server_name}' not found or has no tools.")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Usage: /mcp <server_name>")
            continue

        if user_input.startswith("/call"):
            parts = user_input.split()
            if len(parts) >= 3:
                server_name = parts[1]
                tool_name = parts[2]
                try:
                    result = await execute_mcp_tool(server_name, tool_name, {})
                    print(f"\n🔧 Result from {server_name}.{tool_name}:")
                    print(result)
                except Exception as e:
                    print(f"Error calling tool: {e}")
            else:
                print("Usage: /call <server_name> <tool_name>")
            continue

        # Handle regular chat with simple tool detection
        print_thinking()
        try:
            # Build a description of available tools for the LLM
            tools_description = "Available tools:\n"
            for tool in tools:
                schema = getattr(tool, 'args_schema', {})
                tools_description += f"- {tool.name}: {tool.description}\n"
                tools_description += f"  Arguments schema: {schema}\n"
            
            # First, ask the LLM if it needs to use any tools and how
            planning_prompt = f"""User question: {user_input}

{tools_description}

Based on the user's question, do you need to use any of the available tools to get information before answering? 

If yes, respond with:
TOOL_NEEDED: <tool_name>
ARGUMENTS: <json_arguments>
REASON: <why this tool is needed>

If no tools are needed, respond with:
NO_TOOLS_NEEDED

Only suggest using a tool if it's clearly necessary to answer the user's question."""

            planning_response = await chat_model.ainvoke(planning_prompt)
            planning_text = planning_response.content.strip()
            
            if planning_text.startswith("TOOL_NEEDED:"):
                # Parse the LLM's tool request
                lines = planning_text.split('\n')
                tool_name = lines[0].replace("TOOL_NEEDED:", "").strip()
                arguments_line = next((line for line in lines if line.startswith("ARGUMENTS:")), "")
                reason_line = next((line for line in lines if line.startswith("REASON:")), "")
                
                if arguments_line:
                    import json
                    try:
                        args_text = arguments_line.replace("ARGUMENTS:", "").strip()
                        tool_args = json.loads(args_text)
                        
                        # Find and execute the requested tool
                        target_tool = None
                        for tool in tools:
                            if tool.name == tool_name:
                                target_tool = tool
                                break
                        
                        if target_tool:
                            print(f"\n🔧 TOOL CALL")
                            print(f"Tool Name: {tool_name}")
                            print(f"Tool Input: {tool_args}")
                            print(f"Reason: {reason_line.replace('REASON:', '').strip()}")
                            print("Executing...")
                            
                            # Execute the tool with proper async handling
                            try:
                                import concurrent.futures
                                import asyncio
                                
                                # Create a new event loop in a thread pool to avoid conflicts
                                def run_tool_in_new_loop():
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    try:
                                        return new_loop.run_until_complete(target_tool.ainvoke(tool_args))
                                    finally:
                                        new_loop.close()
                                
                                # Run the tool in a separate thread
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(run_tool_in_new_loop)
                                    tool_result = future.result(timeout=30)  # 30 second timeout
                                
                                # Log the tool output
                                print(f"\n📤 TOOL OUTPUT")
                                print(f"Raw Result: {tool_result}")
                                print("─" * 50)
                                    
                            except Exception as tool_error:
                                print(f"❌ Tool execution error: {tool_error}")
                                print("Let me try to answer without using tools.")
                                response = await chat_model.ainvoke(user_input)
                                print(response.content)
                                continue
                            
                            # Now ask the LLM to provide the final answer using the tool result
                            final_prompt = f"""User asked: {user_input}

I used the tool '{tool_name}' with arguments: {tool_args}

The tool returned this result:
{tool_result}

Please provide a helpful, natural language response to the user's original question based on this information."""
                            
                            final_response = await chat_model.ainvoke(final_prompt)
                            print(f"\n{final_response.content}")
                        else:
                            print(f"❌ Tool '{tool_name}' not found")
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing tool arguments: {e}")
                        print("Let me try to answer without using tools.")
                        response = await chat_model.ainvoke(user_input)
                        print(response.content)
                else:
                    print("❌ No arguments provided for tool")
            else:
                # No tools needed, just use the LLM directly
                response = await chat_model.ainvoke(user_input)
                print(response.content)
                
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    # Cleanup
    print("\n🧹 Cleaning up...")
    await cleanup_mcp()
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main()) 