# MCP Agent Chatbot

An intelligent chatbot powered by LangChain Agents that can reason through complex problems and chain multiple tool calls together to provide comprehensive answers.

## Features

- **ReAct Agent**: Uses the ReAct (Reasoning + Acting) pattern for intelligent decision making
- **Tool Chaining**: Can automatically chain multiple tool calls together
- **Memory**: Remembers conversation history within sessions
- **Streaming**: Real-time streaming of agent responses and tool calls
- **MCP Integration**: Seamless integration with Model Context Protocol servers

## Key Improvements

### Before (Simple Implementation)
- Basic tool detection with manual prompting
- Single tool calls only
- No conversation memory
- Limited reasoning capabilities

### After (Agent-Powered)
- Intelligent reasoning with ReAct pattern
- Automatic tool chaining and complex workflows  
- Persistent conversation memory
- Advanced error handling and recovery
- Real-time streaming of thought process

## Usage

### Basic Commands
- `/tools` - List available MCP tools
- `/memory` - Show current conversation thread
- `/new` - Start a new conversation (clear memory)
- `/help` - Show help message
- `exit` - Exit the chatbot

### Example Interactions

**Simple Query:**
```
You > What tables are in the database?
Agent > I'll help you check what tables are available in the database.

🔧 Using tool: postgres_list_tables
📥 Input: {}

Based on the database query, here are the tables available:
- users: Contains user account information
- orders: Contains order records
- products: Contains product catalog
```

**Complex Reasoning:**
```
You > Show me the top 5 customers by total order value, but only include customers who have made more than 3 orders
Agent > I'll help you find the top customers by total order value with more than 3 orders. Let me break this down:

🔧 Using tool: postgres_execute_sql
📥 Input: {
  "sql": "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id HAVING COUNT(*) > 3"
}

🔧 Using tool: postgres_execute_sql  
📥 Input: {
  "sql": "SELECT c.customer_name, SUM(o.total_amount) as total_value FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.id IN (...) GROUP BY c.id, c.customer_name ORDER BY total_value DESC LIMIT 5"
}

Here are the top 5 customers by total order value (with more than 3 orders):
1. John Smith - $2,450.00 (5 orders)
2. Mary Johnson - $1,890.00 (4 orders)
...
```

## Architecture

The chatbot uses a modular architecture:

1. **MCPToolWrapper**: Converts MCP tools to LangChain-compatible tools
2. **ReAct Agent**: Handles reasoning and tool orchestration  
3. **Memory System**: Maintains conversation history
4. **Streaming Handler**: Provides real-time feedback

## Running the Chatbot

```bash
cd mcp_chatbot
python src/main.py
```

Make sure you have:
- Docker running (for MCP servers)
- Ollama running with your configured model
- Environment variables set (OLLAMA_BASE_URL, OLLAMA_MODEL) 