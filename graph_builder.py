from state import GraphState
from langgraph.graph import StateGraph, START, END
from agent_factory import AgentFactory
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from LLMFactory import LLMFactory
from fastapi import FastAPI, APIRouter
import uvicorn
import argparse
import os
from langgraph.types import Command, interrupt
from typing import Literal
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from tools.common import check_authentication_status
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage # For type checking in response
from dotenv import load_dotenv

load_dotenv()

def user_input_node(
    state: GraphState, config
) -> Command[Literal["supervisor"]]:
    """A node for collecting user input."""

    user_prompt = state.user_prompt
    user_input = interrupt(value=user_prompt)
    active_agent = state.last_active_agent

    return Command(
        update={
            "user_response": user_input,
        },
        goto=active_agent,
    )

async def build_graph():

    builder = StateGraph(GraphState)

    idv_agent = await AgentFactory.create_agent("idv")
    items_agent = await AgentFactory.create_agent("items")

    supervisor = create_supervisor(
        agents=[idv_agent, items_agent],
        model=LLMFactory.get_llm("small_llm"),
        state_schema=GraphState,
        tools=[check_authentication_status],
        prompt=(
            "call the check_authentication_status tool and then Greet the user and say hello"
            "if the user is not authenticated, call the idv_agent"
            "if the user is authenticated and user wants to see their items, call the items_agent"
        )
    ).compile()

    tool_node = ToolNode([check_authentication_status])

    builder.add_node("supervisor",  supervisor)
    builder.add_node("idv_agent", idv_agent)
    builder.add_node("items_agent", items_agent)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "idv_agent")
    builder.add_edge("supervisor", "items_agent")
    builder.add_edge("idv_agent", "tool_node")
    builder.add_edge("items_agent", "tool_node")

    builder.add_edge("supervisor", END)
    
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# FastAPI app and router
app = FastAPI(
    title="Contact Center Server",
    description="API endpoints for channel adapters to communicate with the conversation controller",
    version="1.0.0",
)
router = APIRouter()

@app.on_event("startup")
async def startup_event():
    print("in startup event")

@router.post("/messaging", response_class=JSONResponse)
async def handle_messaging(request: Request) -> Response:

    body = await request.body()
    body_str = body.decode('utf-8')

    print(body_str)
    graph = await build_graph()

    thread_id = f"thread_12345678"
    # Increase recursion limit to allow for the enrollment->welcome->prompt sequence
    thread_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15, # Or another number >= 5
        "metadata": {
           
        }
    }

    msg = {
        "messages": [HumanMessage(content="Hi")] # Use HumanMessage
    }

    print(msg)  

    response = await graph.ainvoke(msg, thread_config)
    print("Full graph response state:", response)

    # Extract relevant information for the HTTP response
    final_messages = response.get("messages", [])
    last_ai_message_content = "No AI message found."
    
    # Iterate backwards to find the last AIMessage
    for message in reversed(final_messages):
        if isinstance(message, AIMessage):
            last_ai_message_content = message.content
            break

    return {
        "is_authenticated": response.get("is_authenticated"),
        "last_response_from_state": response.get("last_response"),
        "final_ai_message": last_ai_message_content,
        "all_messages_in_state": [
            {"role": msg.type, "content": msg.content} for msg in final_messages # Serialize for JSON
        ]
    }

app.include_router(router)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the Contact Center Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")),
                        help="Port to run the server on (default: 8000 or PORT env variable)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "debug"),
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Set the logging level (default: info)")
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Log to file when set",
    )

    args = parser.parse_args()
    # cli_args = args # Store args for lifespan -- REMOVED

    # Set environment variables based on parsed args BEFORE starting Uvicorn
    os.environ["APP_LOG_LEVEL"] = args.log_level # Already uppercase from parser
    os.environ["APP_LOG_TO_FILE"] = str(args.log_to_file)

    # Use print for the initial message before Uvicorn's logging takes over
    print(f"Attempting to start Contact Center Server on {args.host}:{args.port} with log level {args.log_level}")

    uvicorn.run(
        "graph_builder:app",
        host=args.host, 
        port=args.port, 
        reload=args.reload, 
        log_level=args.log_level.lower(),
        proxy_headers=True
    ) 
