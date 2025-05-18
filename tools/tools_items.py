from langchain_core.tools import tool
from typing import Annotated
from langgraph.prebuilt import InjectedState
from state import GraphState

items = [
        {
            "item_id": "111",
            "item_name": "Item 1",
            "item_description": "Item 1 description",
            "item_price": 100,
            "status": "available"
        },
        {
            "item_id": "222",
            "item_name": "Item 2",
            "item_description": "Item 2 description",
            "item_price": 200,
            "status": "available"
        }
    ]

@tool
async def lookup_items(state: Annotated[GraphState, InjectedState]):
    """Lookup all items by customer id"""

    customer_id = state.authentication_state.customer_id
    print(f"Looking up items for customer {customer_id}")
    return items

@tool
async def update_item(state: Annotated[GraphState, InjectedState], item_id: str, status: str):
    """Update an item by customer id and item id"""

    customer_id = state.authentication_state.customer_id
    print(f"Updating item {item_id} for customer {customer_id} to {status}")

    return {
        "updated": True
    }
