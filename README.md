## Start Postgres

```bash
docker-compose up -d
```

```mermaid
Read more about integrating this database with a Dockerized MCP server on my website [here](https://miketoscano.com/blog/?post=docker-mcp-toolkit-postgres)

graph TB
    %% Core Components
    subgraph UserLayer["👥 User Interface Layer"]
        ChatInterface[💬 Chat Application<br/>Python/Streamlit/FastAPI]
    end
    
    subgraph AgentLayer["🤖 AI Agent Layer"]
        LangChainAgent[⚡ LangChain ReAct Agent<br/>- Query Planning<br/>- Multi-step Reasoning<br/>- Tool Orchestration]
        AgentMemory[💾 Conversation State<br/>MemorySaver]
    end
    
    subgraph MCPLayer["🔗 MCP Integration Layer"]
        MCPClient[📡 MCP Client<br/>langchain-mcp-adapters]
        MCPServer[🐳 MCP Server<br/>Docker Container<br/>mcp/postgres:latest]
    end
    
    subgraph DataLayer["📊 Data Layer"]
        Database[(🗄️ PostgreSQL Database<br/>Your Business Data)]
    end
    
    subgraph Infrastructure["🏗️ Infrastructure"]
        Docker[🐳 Docker Compose<br/>- Container Orchestration<br/>- Network Isolation<br/>- Service Discovery]
        LLMRuntime[🧠 LLM Runtime<br/>Ollama/OpenAI API]
    end
    
    %% Connections
    ChatInterface --> LangChainAgent
    LangChainAgent --> AgentMemory
    LangChainAgent --> MCPClient
    MCPClient -->|stdio protocol| MCPServer
    MCPServer -->|SQL queries| Database
    LangChainAgent --> LLMRuntime
    
    %% Infrastructure connections
    ChatInterface -.->|deployed in| Docker
    MCPServer -.->|managed by| Docker
    Database -.->|managed by| Docker
    
    %% Key Technologies Stack
    subgraph TechStack["🔧 Technology Stack"]
        direction LR
        Backend["Python 3.11+<br/>LangChain<br/>Pydantic"]
        Protocol["MCP Protocol<br/>JSON-RPC<br/>stdio transport"]
        Database_Tech["PostgreSQL<br/>Connection pooling<br/>Read-only access"]
        Deployment["Docker Compose<br/>Container networking<br/>Volume persistence"]
    end
    
    %% Security & Governance
    subgraph Security["🔒 Security Model"]
        direction TB
        Isolation[🛡️ Process Isolation<br/>Each query in container]
        ReadOnly[📖 Read-only Access<br/>No DDL/DML operations]
        Validation[✅ Query Validation<br/>MCP server enforces limits]
    end
    
    %% Data Flow
    Database -->|results| MCPServer
    MCPServer -->|JSON response| MCPClient
    MCPClient -->|tool results| LangChainAgent
    LangChainAgent -->|formatted output| ChatInterface
    
    %% Styling
    classDef userLayer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef agentLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef mcpLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef dataLayer fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef infraLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef techLayer fill:#f5f5f5,stroke:#616161,stroke-width:2px
    
    class UserLayer,ChatInterface userLayer
    class AgentLayer,LangChainAgent,AgentMemory agentLayer
    class MCPLayer,MCPClient,MCPServer mcpLayer
    class DataLayer,Database dataLayer
    class Infrastructure,Docker,LLMRuntime infraLayer
    class TechStack,Security techLayer
```