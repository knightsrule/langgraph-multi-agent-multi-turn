from langchain_core.tools import tool
from state import GraphState
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool

@tool
def check_authentication_status(input_not_used: str = "") -> dict:
    """Checks the current authentication status and returns it."""
    print("Tool 'check_authentication_status' called by supervisor")
    # In a real scenario, you'd check a session, database, etc.
    # For this test, we can hardcode it.
    return {"is_authenticated": True, "last_response": "User has been successfully authenticated via tool."}

#def update_intent(intent: str, tool_call_id: Annotated[str, InjectedToolCallId], state: Annotated[State, InjectedState]) -> Command:
#    """Update the intent"""
#    print(f"Updating intent to {intent} from update_intent tool")
#    return Command(update={
#        "intent": intent,
#        "messages": [
#            ToolMessage(content=f"Intent updated to {intent}.", tool_call_id=tool_call_id),
#            AIMessage(content=f"Intent updated to {intent}.")
#        ]
#    })