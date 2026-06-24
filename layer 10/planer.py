
from agentStates import AgentStates
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from Schemas import calculate , get_weather , ToolRegistry , MathSchema,WeatherSchema
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple


def mock_llm_planner(state: AgentStates) -> Tuple[str, str, Any, float]:
    """
    Simulates an LLM analyzing the state and deciding the next action.
    In production, this translates state.messages into a prompt and calls an API.
    """
    simulated_cost = 0.01

    if state.step_count == 0:
        return "Tool Call", "get_weather", {"location": "San Francisco"}, simulated_cost
    elif state.step_count == 1:
        return "Tool Call", "calculate", {"expression": "72 * 2"}, simulated_cost
    else:
        return "Final Answer", None, "The weather is 72F, and double that is 144.", simulated_cost

def run_verifier(final_answer: str, goal: str) -> bool:
    """
    Checks if the final output actually satisfies the prompt.
    Could be a deterministic check or an 'LLM-as-a-judge' call.
    """
    if "144" in final_answer:
        return True
    return False

# ==========================================
# 4. CORE EXECUTOR
# ==========================================
def run_agent(state: AgentStates, registry: ToolRegistry):
    trace_log = [f"Goal: {state.user_goal}"]
    
    while state.step_count < state.max_steps and state.current_cost < state.max_budget:
        # Check safety gates
        if state.requires_approval:
            trace_log.append("SYSTEM ALARM: Agent paused for human approval.")
            break
            
        # Planner decides next action
        action, tool_name, tool_args, cost = mock_llm_planner(state)
        
        # Update system limits
        state.step_count += 1
        state.current_cost += cost
        trace_log.append(f"Step {state.step_count} (Cost: ${state.current_cost:.2f}): Decided on {action}")

        # Execute
        if action == "Final Answer":
            is_valid = run_verifier(tool_args, state.user_goal)
            trace_log.append(f"Verification Check: {'Passed' if is_valid else 'Failed'}")
            return tool_args, trace_log
            
        elif action == "Tool Call":
            result = registry.validate_and_execute(tool_name, tool_args)
            state.messages.append({"role": "system", "content": result})
            trace_log.append(f"-> Executed '{tool_name}': {result}")
            
    # System Failure Modes
    if state.step_count >= state.max_steps:
        trace_log.append("SYSTEM HALT: Hit maximum step limit.")
    if state.current_cost >= state.max_budget:
        trace_log.append("SYSTEM HALT: Hit maximum budget limit.")
        
    return "Agent failed safely.", trace_log

# ==========================================
# 5. EXECUTION BOOTSTRAP
# ==========================================
if __name__ == "__main__":
    # Setup Registry
    registry = ToolRegistry()
    registry.register("get_weather", WeatherSchema, get_weather)
    registry.register("calculate", MathSchema, calculate)

    # Initialize State
    initial_state = AgentStates(
        user_goal="Get the weather in SF and double the temperature."
    )

    # Run
    final_output, traces = run_agent(initial_state, registry)

    print("\n--- AGENT TRACE LOG ---")
    for trace in traces:
        print(trace)
    
    print("\n--- FINAL OUTPUT ---")
    print(final_output)