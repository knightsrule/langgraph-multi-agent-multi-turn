from pydantic import BaseModel, Field
import os
from langgraph.managed import RemainingSteps
from langchain_core.messages import BaseMessage # Add this import

class AutenticationState(BaseModel):
    pending_otp: str = Field(default="")
    pending_customer_id: str = Field(default="")

class GraphState(BaseModel):
    """Overall state passed between LangGraph invocations."""

    customer_id: str = Field(default="")
    last_response: str = Field(default="")
    is_authenticated: bool = Field(default=False)
    
    #last_response: str = Field(default="")

    authentication_state: AutenticationState = Field(default_factory=AutenticationState)
    #last_active_agent: str = Field(default="")
    #user_prompt: str = Field(default="")
    
    messages: list[BaseMessage] = Field(default_factory=list) # Changed from list[dict]
    remaining_steps: RemainingSteps = Field(default_factory=RemainingSteps)