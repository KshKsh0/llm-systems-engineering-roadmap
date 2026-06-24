from pydantic import BaseModel

class ChatRequest(BaseModel):
    model: str = "custom-llama"
    messages: list[dict]
    max_tokens: int = 50
    stream: bool = True