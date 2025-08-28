# 🤖 Deep Agent System

A comprehensive deep agent system for Oracle database operations with AWS Bedrock integration, implementing the four key components of deep agents: **Planning Tool**, **Sub Agents**, **File System Access**, and **Detailed Prompts**.

## 🎯 Overview

This system creates an intelligent agent that can:
- **Communicate** with users through natural language queries
- **Fetch results** from Oracle databases with smart table/schema selection
- **Choose intelligently** over what schema or table to pick from
- **Maintain memory** to help with context and learning
- **Detect discrepancies** in data (business logic ready for implementation)

## 🏗️ Architecture

### Four Key Components

1. **📋 Planning Tool** (`PlannerAgent`)
   - Analyzes complex user queries
   - Breaks down operations into logical steps
   - Determines required tables and schemas
   - Coordinates sub-agent execution

2. **🔧 Sub Agents**
   - **QueryAgent**: Intelligent SQL generation and execution
   - **DataQualityAgent**: Data quality analysis and discrepancy detection
   - **DMLAgent**: INSERT, UPDATE, DELETE operations
   - **BaseAgent**: Common functionality and memory management

3. **💾 File System Access**
   - Persistent storage of queries and results
   - Query caching for performance
   - Agent memory persistence
   - Structured data organization

4. **🎯 Detailed Prompts**
   - Comprehensive system prompts for each agent
   - Context-aware AI interactions
   - Business logic integration ready

## 🚀 Features

- **Intelligent Query Planning**: AI-powered query analysis and execution planning
- **Smart Table Selection**: Automatic selection of relevant tables and schemas
- **Data Quality Analysis**: Comprehensive data quality assessment and recommendations
- **Memory Management**: Redis-based memory system for context retention
- **File System Integration**: Persistent storage and caching of all operations
- **Multi-Agent Coordination**: Seamless coordination between specialized agents
- **Oracle Database Integration**: Full Oracle database support with connection pooling
- **AWS Bedrock Integration**: Advanced AI capabilities through Claude models

## 📋 Prerequisites

- Python 3.8+
- Oracle Database (with cx_Oracle)
- Redis Server (for memory management)
- AWS Account with Bedrock access
- AWS Session Token (for temporary credentials)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agent-code
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your actual values
   ```

4. **Configure your environment**
   - Set Oracle database credentials
   - Configure AWS credentials and session token
   - Set Redis connection details
   - Adjust agent parameters as needed

## ⚙️ Configuration

### Environment Variables

```bash
# Oracle Database
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE=XE
ORACLE_USER=your_username
ORACLE_PASSWORD=your_password

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_SESSION_TOKEN=your_session_token
AWS_REGION=us-east-1

# Bedrock Model
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Agent Settings
AGENT_MEMORY_TTL=3600
MAX_TOKENS=4000
TEMPERATURE=0.1
```

## 🎮 Usage

### Interactive Mode
```bash
python main.py interactive
```

### Demo Mode
```bash
python main.py demo
```

### Available Commands
- `help` - Show available commands
- `status` - System status and health
- `history` - Query and execution history
- `clear` - Clear all caches
- `quality <table_name>` - Analyze data quality
- `query <sql>` - Execute direct SQL
- Natural language queries - Process through deep agent

## 🔍 Example Queries

### Natural Language Queries
```
"Show me all customers from New York"
"What is the data quality of the ORDERS table?"
"Find all products with price greater than $100"
"Analyze the structure of the INVENTORY table"
```

### Data Quality Analysis
```
quality CUSTOMERS
quality ORDERS
quality PRODUCTS
```

### Direct SQL Execution
```
query SELECT COUNT(*) FROM CUSTOMERS
query SELECT * FROM ORDERS WHERE ORDER_DATE > SYSDATE - 30
```

## 🏗️ System Components

### Core Agents

#### PlannerAgent
- **Purpose**: Coordinates complex operations and creates execution plans
- **Capabilities**: Query analysis, task breakdown, sub-agent coordination
- **Memory**: Stores execution plans and validation feedback

#### QueryAgent
- **Purpose**: Generates and executes SQL queries intelligently
- **Capabilities**: Query intent analysis, table selection, SQL generation
- **Memory**: Caches query analysis and generated SQL

#### DataQualityAgent
- **Purpose**: Analyzes data quality and detects discrepancies
- **Capabilities**: Schema analysis, data validation, quality scoring
- **Memory**: Stores quality analysis results and recommendations

#### DMLAgent
- **Purpose**: Handles data modification operations
- **Capabilities**: INSERT, UPDATE, DELETE with validation
- **Memory**: Tracks DML operations and validation results

### Database Integration

- **Connection Pooling**: Efficient Oracle database connections
- **Schema Discovery**: Automatic table and column information retrieval
- **Query Execution**: Safe and parameterized query execution
- **Transaction Management**: Support for both auto and manual transactions

### Memory System

- **Redis Integration**: Fast in-memory storage for agent memory
- **Persistent Storage**: File-based storage for queries and results
- **Context Retention**: Maintains conversation and operation context
- **Cache Management**: Intelligent caching with TTL support

## 🔧 Customization

### Adding Business Logic

The system is designed to be easily extensible. To add your business logic for discrepancy detection:

1. **Extend DataQualityAgent**
   ```python
   def detect_discrepancies(self, table_name: str, business_rules: Dict[str, Any]):
       # Implement your business logic here
       pass
   ```

2. **Define Business Rules**
   ```python
   business_rules = {
       "range_check": {
           "type": "range_check",
           "column": "PRICE",
           "min": 0,
           "max": 10000
       },
       "uniqueness_check": {
           "type": "uniqueness_check",
           "column": "PRODUCT_ID"
       }
   }
   ```

3. **Integrate with Planning**
   ```python
   # The planner will automatically include data quality checks
   # when business rules are provided
   ```

### Adding New Agents

1. **Create Agent Class**
   ```python
   from .base_agent import BaseAgent
   
   class CustomAgent(BaseAgent):
       def __init__(self):
           super().__init__("Custom")
           self.system_prompt = self._get_custom_prompt()
       
       def process(self, input_data):
           # Implement your logic
           pass
   ```

2. **Register with DeepAgent**
   ```python
   self.custom_agent = CustomAgent()
   self.agents["custom"] = self.custom_agent
   ```

## 📊 Monitoring and Debugging

### System Status
```bash
status
```
Shows overall system health, database connectivity, and agent status.

### Query History
```bash
history
```
Displays all processed queries and execution results.

### Agent Memory
Each agent maintains its own memory for debugging and analysis.

### Logging
Comprehensive logging at INFO level with detailed error tracking.

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify Oracle credentials and network connectivity
   - Check if Oracle service is running
   - Ensure cx_Oracle is properly installed

2. **AWS Bedrock Access Denied**
   - Verify AWS credentials and session token
   - Check Bedrock permissions
   - Ensure region is correct

3. **Redis Connection Failed**
   - Verify Redis server is running
   - Check Redis connection parameters
   - System will fall back to file-based memory if Redis unavailable

4. **Memory Issues**
   - Check available disk space
   - Verify directory permissions
   - Use `clear` command to free up cache

### Debug Mode

Enable detailed logging by modifying the logging level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

- **Multi-Database Support**: Extend beyond Oracle to other databases
- **Advanced Analytics**: Integrate with data science libraries
- **API Interface**: REST API for external integrations
- **Web Dashboard**: Web-based monitoring and control interface
- **Machine Learning**: Predictive analytics and anomaly detection
- **Workflow Automation**: Complex multi-step process automation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support and questions, please open an issue in the repository.

---

**Built with ❤️ using Python, Oracle, AWS Bedrock, and Redis** 