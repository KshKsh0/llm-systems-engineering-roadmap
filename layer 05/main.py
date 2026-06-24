from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from schema import ChatRequest
import asyncio
import uvicorn

app = FastAPI(title="LLM Systems Engineer - Mock Serving Engine")


#  the values here the LLMs since it's not optimal to do that in python anyways 
async def mock_generation_stream(prompt_length: int, max_tokens: int):
    """
    Simulates the physical constraints of GPU inference.
    """
    
    # ---------------------------------------------------------
    # 1. PREFILL PHASE (Determines TTFT)
    # The longer the prompt, the longer the initial KV cache setup.
    # ---------------------------------------------------------
    prefill_latency = 0.005 * prompt_length  
    await asyncio.sleep(prefill_latency)     
    
    
    yield "data: {\"choices\": [{\"delta\": {\"role\": \"assistant\", \"content\": \"\"}}]}\n\n"

    # ---------------------------------------------------------
    # 2. DECODE PHASE (Determines TPOT)
    # Generating one token at a time, constrained by memory bandwidth.
    # ---------------------------------------------------------
    tpot_latency = 0.04  # e.g., 40ms per output token (25 tokens/sec)
    
    for i in range(max_tokens):
        await asyncio.sleep(tpot_latency)   
        chunk = f"token_{i} "
        yield f"data: {{\"choices\": [{{\"delta\": {{\"content\": \"{chunk}\"}}}}]}}\n\n"
        
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def completions_endpoint(request: ChatRequest):
    
    prompt_text = str(request.messages)
    estimated_input_tokens = len(prompt_text.split())
    
    if request.stream:
       
        return StreamingResponse(
            mock_generation_stream(estimated_input_tokens, request.max_tokens),
            media_type="text/event-stream"
        )
    else:
        return {"error": "This systems demo requires stream=True to demonstrate TTFT vs TPOT"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)