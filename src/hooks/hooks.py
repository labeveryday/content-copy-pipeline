# Hooks for logging tool invocations
from strands.hooks import HookProvider, HookRegistry
from strands.experimental.hooks import BeforeToolInvocationEvent


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