from pydantic import BaseModel
from typing import List, Dict, Any, Tuple

class WeatherSchema(BaseModel):
    location:str

class MathSchema(BaseModel):
    expression:str 


def get_weather(location: str) -> str:
    # Simulated API call
    return f"The weather in {location} is 72F."

def calculate(expression: str) -> str:
    # Simulated calculator tool
    try:
        return str(eval(expression))
    except Exception:
        return "Error: Invalid math expression"
    
    
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, schema: BaseModel, func: callable):
        self.tools[name] = {"schema": schema, "func": func}

    def validate_and_execute(self, name: str, args: dict):
        if name not in self.tools:
            return "Error: Tool not found."
        try:
           
            validated_args = self.tools[name]["schema"](**args)
            return self.tools[name]["func"](**validated_args.model_dump())
        except Exception as e:
            return f"Error: Invalid arguments - {str(e)}"