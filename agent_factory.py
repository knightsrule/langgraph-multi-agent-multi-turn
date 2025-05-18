from tools.idv_tools import lookup_customer, send_otp, verify_otp
from tools.tools_items import lookup_items, update_item
from langgraph.prebuilt import create_react_agent
from LLMFactory import LLMFactory
from state import GraphState
class AgentFactory:
    
    @staticmethod
    async def _lookup_agent_config(agent_id: str):
        if agent_id == "idv":
            return {
                "model": LLMFactory.get_llm("small_llm"),
                "tools": [lookup_customer, send_otp, verify_otp],
                "prompt": (
                    "You are a user authentication assistant." 
                    "You will prompt the user for their phone number and pin." 
                    "Then, you will validate this information using lookup_customer tool." 
                    "If you find a vaild customer, send a one time passcode using send_otp tool." 
                    "Then validate this otp using verify_otp tool." 
                    "If the otp is valid, return the customer id to the user."
                ),
                "agent_id": agent_id
            }
        elif agent_id == "items":
            return {
                "model": LLMFactory.get_llm("small_llm"),
                "tools": [lookup_items, update_item],
                "prompt": "You are an item assistant. Help the user see and manage their items.",
                "agent_id": agent_id
            }
        else:
            raise ValueError(f"Agent {agent_id} not found")

    @staticmethod
    async def create_agent(agent_id: str):
        config = await AgentFactory._lookup_agent_config(agent_id)
        return create_react_agent(
            name=config["agent_id"],
            model=config["model"],
            tools=config["tools"],
            prompt=config["prompt"],
            state_schema=GraphState
        )
