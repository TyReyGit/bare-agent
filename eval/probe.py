"""Quick tool-call reliability probe for the configured model.

Runs one simple task three times and reports retries and outcomes. Cheap
enough to check quota and model behaviour without burning a full eval run.
"""

from bare_agent.agent import run_agent

TASK = "Read notes.txt from the workspace and tell me how many lines it contains."

for i in range(1, 4):
    print(f"\n########## RUN {i} ##########")
    try:
        answer = run_agent(TASK)
        print(f">>> OK: {answer}")
    except Exception as exc:
        print(f">>> ERROR: {type(exc).__name__}: {exc}")
