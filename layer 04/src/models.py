import time
import requests
from typing import Tuple, Dict, Any


class LocalModelClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.api_url = f"{host}/api/generate"
        
        
        
    def _call_ollama(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Helper to make a POST request to the local Ollama instance."""
        payload ={
            'model' : model_name,
            'prompt':prompt,
            "stream":False,
            'options':{
                'temperature': 0.0 # greedy search no creativty
            }    
            }
        
        try:
            respones =requests.post(self.api_url ,   json =payload)
            respones.raise_for_status()
            return respones.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to reach local Ollama server: {e}")
        
    def query_baseline(self, prompt: str) -> Tuple[str, float, int]:
        """
        Queries the standard instruction-tuned model.
        Returns: (final_answer, latency_seconds, token_count)
        """
        start_time = time.time()
        data = self._call_ollama("llama3:8b", prompt)
        latency = time.time() - start_time
    
        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        answer = data.get("response", "").strip()
        
        return answer, latency, tokens
            
    def query_reasoning(self, prompt: str) -> Tuple[str, str, float, int]:
        """
        Queries the reasoning model.
        Returns: (thinking_process, final_answer, latency_seconds, token_count)
        """
        start_time = time.time()
        data = self._call_ollama("deepseek-r1:8b", prompt)
        latency = time.time() - start_time
        
        raw_response = data.get("response", "")
        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        
   
        thinking = ""
        final_answer = raw_response
        
        if "<think>" in raw_response and "</think>" in raw_response:
            parts = raw_response.split("</think>")
            thinking = parts[0].replace("<think>", "").strip()
            final_answer = parts[1].strip()
            
        return thinking, final_answer, latency, tokens