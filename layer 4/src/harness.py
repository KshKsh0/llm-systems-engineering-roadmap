from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Dict
from src.models import LocalModelClient
from src.tools import ToolBox

@dataclass
class EvalResult:
    task_id: str
    method: Literal["baseline", "reasoning", "tool_verified"]
    latency_seconds: float
    token_cost: int
    correctness: bool
    final_answer: str
    thinking_process: Optional[str] = None
    tool_usage: Optional[str] = None

class ReasoningEvalHarness:
    def __init__(self, client: LocalModelClient, tools: ToolBox):
        self.client = client
        self.tools = tools

    def check_correctness(self, expected: str, actual: str) -> bool:
        """A simple substring evaluator. In production, use an LLM-as-a-judge."""
        return expected.lower() in actual.lower()

    def run_baseline(self, task_id: str, question: str, expected: str) -> EvalResult:
        prompt = f"Answer the following question directly. Question: {question}"
        answer, latency, tokens = self.client.query_baseline(prompt)
        
        return EvalResult(
            task_id=task_id,
            method="baseline",
            latency_seconds=round(latency, 2),
            token_cost=tokens,
            correctness=self.check_correctness(expected, answer),
            final_answer=answer
        )

    def run_reasoning(self, task_id: str, question: str, expected: str) -> EvalResult:
        prompt = f"Think through this step-by-step, then answer. Question: {question}"
        thinking, answer, latency, tokens = self.client.query_reasoning(prompt)
        
        return EvalResult(
            task_id=task_id,
            method="reasoning",
            latency_seconds=round(latency, 2),
            token_cost=tokens,
            correctness=self.check_correctness(expected, answer),
            final_answer=answer,
            thinking_process=thinking
        )

    def run_tool_verified(self,task_id:str ,question:str ,expected:str )->EvalResult:
        """Simulates a basic ReAct (Reason + Act) loop using the baseline model."""
        
        extract_prompt= f"To solve this: '{question}', what single math equation should I run? Output ONLY the equation (e.g., 15-8). If no math is needed, output 'NONE'."
        equation , lat_1 ,tok_1 = self.client.query_baseline(extract_prompt)
        tool_log = "No tool used."
        calc_result = ""
        if "NONE" not in equation.upper():
            calc_result = self.tools.safe_calculator(equation)
            tool_log = f"Calculated [{equation}] -> {calc_result}"
        
        final_prompt = f"Question: {question}\nTool Result: {calc_result}\nProvide the final answer:"
        answer , lat_2 ,tok_2 = self.client.query_baseline(extract_prompt)
        return EvalResult(
            task_id=task_id,
            method="tool_verified",
            latency_seconds=round(lat_1 + lat_2, 2),
            token_cost=tok_1 + tok_2,
            correctness=self.check_correctness(expected, answer),
            final_answer=answer,
            tool_usage=tool_log
        )
            
    def evaluate_task(self, task: Dict) -> List[Dict]:
        """Runs a single task through all three pipelines."""
        task_id = task["id"]
        q = task["question"]
        ans = task["expected_answer"]
        
        results = [
            asdict(self.run_baseline(task_id, q, ans)),
            asdict(self.run_reasoning(task_id, q, ans)),
            asdict(self.run_tool_verified(task_id, q, ans))
        ]
        return results