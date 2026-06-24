from pydantic import BaseModel, Field
from typing import List, Dict, Any

class AgentStates(BaseModel):
    user_goal :str 
    messages: List[Dict[str, Any]] = []
    current_plan :List[str] = []
    step_count :int =0
    max_steps :int = 5
    current_cost :int = 0
    max_budget :int = .10
    requires_approval:bool = False 
    