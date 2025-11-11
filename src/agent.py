"""
Demo Agent

This agent is a demo of the Strands Agents framework.
It uses the Anthropic claude-haiku-4-5-20251001 model to generate responses.
It also uses the shell, editor, and current_time tools to interact with the system.
It also uses the SlidingWindowConversationManager to manage the conversation history.
It also uses the FileSessionManager to manage the session history.
It also uses the LoggingHook to log the tool invocations.
It also uses the pprint to pretty print the responses.
"""

from pathlib import Path
from datetime import datetime
from strands import Agent
from models import anthropic_model
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager

from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient

# Hooks for logging tool invocations
from strands.hooks import HookProvider, HookRegistry
from strands.experimental.hooks import BeforeToolInvocationEvent

from strands_tools import shell, editor, current_time
from dotenv import load_dotenv
from pprint import pprint


# Load environmental variables
load_dotenv()

# import model
MODEL = anthropic_model(model_id="claude-sonnet-4-5-20250929")

# Create a consistent session ID
# Configure session ID based on the current date
SESSION_ID = datetime.now().strftime("%Y%m%d")

# Create sessions directory
SESSION_DIR = Path("./sessions")
SESSION_DIR.mkdir(exist_ok=True)

# Create session manager
session_manager = FileSessionManager(
    session_id=SESSION_ID,
    storage_dir=str(SESSION_DIR)
)

#Create conversation manager
conversation_manager = SlidingWindowConversationManager(
    window_size=20,
    should_truncate_results=True
)


# Logging hook for tool invocations before they are executed
class LoggingHook(HookProvider):
    def __init__(self):
        self.calls = 0

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolInvocationEvent, self.log_start)

    def log_start(self, event: BeforeToolInvocationEvent) -> None:
        self.calls += 1
        print('='* 60)
        print(f"🔧 TOOL INVOCATION: {self.calls}")
        print('='* 60)
        print(f"Agent: {event.agent.name}")
        print(f"Tool: {event.tool_use['name']}")
        print("Input Parameters:")

        # Pretty print the input with color coding
        import json
        formatted_input = json.dumps(event.tool_use['input'], indent=2)
        for line in formatted_input.split('\n'):
            print(f"{line}")

        print('='* 60)


agentcore_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx",
        args=["awslabs.amazon-bedrock-agentcore-mcp-server@latest"],
        autoApprove=[
            "search_agentcore_docs",
            "fetch_agentcore_doc"
        ]
    )
))

strands_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx",
        args=["strands-agents-mcp-server"],
        autoApprove=[
            "search_docs",
            "fetch_doc"
        ]
    )
))


SYSTEM_PROMPT = """
You are a helpful assistant that specializes in building and optimizing AI agents:
- You can use the AgentCore MCP server to get information about the AgentCore platform.
- You can use the Strands MCP server to get information about the Strands platform.
- You can use the Model selector tool to get information about the best model for a task.
- You can use the shell tool to interact with the system.
- You can use the editor tool to edit files.
- You can use the current_time tool to get the current time.
"""


with agentcore_mcp_client, strands_mcp_client:
    # Get the tools from the MCP server
    tools = agentcore_mcp_client.list_tools_sync() + strands_mcp_client.list_tools_sync()

    agent = Agent(
        model=MODEL,
        tools=[shell, current_time, editor] + tools,
        session_manager=session_manager,
        conversation_manager=conversation_manager,
        hooks=[LoggingHook()],
        callback_handler=None,
        name="Demo Agent",
        load_tools_from_directory=True  # Load tools from the tools directory automatically
        )

    print("\nWelcome to the Demo Agent!\n")
    print("- Type 'exit' to quit.")
    print("- Type 'model' to view the model configuration.")
    print("- Type 'metrics' to view the event loop metrics.")
    print("- Type 'name' to view the agent name.")
    print("- Type 'tools' to view the agent tools.\n")

    while True:
        print("-" * 100)
        prompt = input("> ")
        print("-" * 100 + "\n")

        if prompt == "exit" or prompt == "quit":
            break

        if prompt == "model":
            print(f"Model Configuration: {agent.model.config}")
            continue

        if prompt == "metrics":
            print("Event Loop Metrics:")
            pprint(agent.event_loop_metrics)
            continue

        if prompt == "tools":
            print(f"Agent Tools: {agent.tool_names}")
            continue

        if prompt == "name":
            print(f"Agent Name: {agent.name}")
            continue

        response = agent(prompt)

        print(f"Assistant: {response}")
