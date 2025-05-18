from langchain_core.tools import tool
from state import GraphState, AutenticationState
from langgraph.types import Command, interrupt
from typing import Annotated
from langgraph.prebuilt import InjectedState, create_react_agent
from langchain_core.messages import AIMessage # Add this import
@tool
async def lookup_customer(phone_number: str, pin: str):
    """Lookup a customer by phone number and pin"""

    # TODO: In a real implementation, you would check your database here
    # For this example, we'll assume any combination is valid
    
    auth_state = AutenticationState(pending_customer_id="1234567890")
    
    return Command(
        "update_state",
        {
            "authentication_state": auth_state,
            "messages": [AIMessage(content="Customer found! Let's verify your identity.")]
        }
    )

@tool
async def send_otp(phone_number: str, state: Annotated[GraphState, InjectedState]):
    """Send a one-time password to the customer"""

    auth_state = state.authentication_state
    auth_state.pending_otp = "123456"

    #TODO: Update the state's authentication_state with pending otp

    return Command(
        "update_state",
        {
            "authentication_state": auth_state
        }
    )

@tool
async def verify_otp(phone_number: str, otp: str, state: Annotated[GraphState, InjectedState]):
    """Verify a one-time password"""

    auth_state = state.authentication_state

    # If OTP is valid, update the customer_id
    if otp == auth_state.pending_otp:
        # Update the auth state with customer_id indicating successful authentication
        auth_state.customer_id = auth_state.pending_customer_id
        auth_state.pending_otp = None  # Clear the pending OTP after successful verification
        
        return Command(
            "update_state",
            {
                "authentication_state": auth_state,
                "last_active_agent": "supervisor"  # Return control to supervisor after authentication
            }
        )
    else:
        # OTP is invalid
        auth_state.pending_otp = None  # Clear the pending OTP after failed verification
        
        return Command(
            "update_state",
            {
                "authentication_state": auth_state,
                "messages": state.messages + [AIMessage(content="Invalid OTP. Please try again.")]
            }
        )