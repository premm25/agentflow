"""Worker agents. Each module exposes `async def run(ctx, inp) -> AgentTaskResult`.

Agents are plain async functions run in-process by the harness runner, one after another. They
receive only an AgentContext (tools, logs, memory) and never talk to each other.
"""
