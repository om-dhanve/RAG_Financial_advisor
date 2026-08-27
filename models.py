#Defining Pydantic models for input and output validation

from pydantic import BaseModel, Field
from config import settings
from typing import List

class ChatRequest(BaseModel):
    message : str = Field(
        min_length=1,
        max_length=2000,
        description="Chat request to AI Agent"
    )

class ChatResponse(BaseModel):
    response : str
    sources : list[dict]
    context : list
    model_name : str = settings.llm_model
    response_time : float | None = None
    thread_id : str | None = None

class MetricsResponse(BaseModel):
    """Metrics endpoint response"""
    total_requests : int 
    total_errors : int
    error_rate : int 
    token_input_count : int 
    token_output_count : int
    avg_latency_ms : float

class ErrorRespone(BaseModel):
    error : str 
    detail : str | None = None
    request_id : str | None = None