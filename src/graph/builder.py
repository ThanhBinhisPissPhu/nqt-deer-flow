# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# from src.prompts.planner_model import StepType

# from .types import State
# from .nodes import (
#     coordinator_node,
#     planner_node,
#     reporter_node,
#     research_team_node,
#     researcher_node,
#     coder_node,
#     human_feedback_node,
#     background_investigation_node,
# )


# def continue_to_running_research_team(state: State):
#     current_plan = state.get("current_plan")
#     if not current_plan or not current_plan.steps:
#         return "planner"

#     if all(step.execution_res for step in current_plan.steps):
#         return "planner"

#     # Find first incomplete step
#     incomplete_step = None
#     for step in current_plan.steps:
#         if not step.execution_res:
#             incomplete_step = step
#             break

#     if not incomplete_step:
#         return "planner"

#     if incomplete_step.step_type == StepType.RESEARCH:
#         return "researcher"
#     if incomplete_step.step_type == StepType.PROCESSING:
#         return "coder"
#     return "planner"


# def _build_base_graph():
#     """Build and return the base state graph with all nodes and edges."""
#     builder = StateGraph(State)
#     builder.add_edge(START, "coordinator")
#     builder.add_node("coordinator", coordinator_node)
#     builder.add_node("background_investigator", background_investigation_node)
#     builder.add_node("planner", planner_node)
#     builder.add_node("reporter", reporter_node)
#     builder.add_node("research_team", research_team_node)
#     builder.add_node("researcher", researcher_node)
#     builder.add_node("coder", coder_node)
#     builder.add_node("human_feedback", human_feedback_node)
#     builder.add_edge("background_investigator", "planner")
#     builder.add_conditional_edges(
#         "research_team",
#         continue_to_running_research_team,
#         ["planner", "researcher", "coder"],
#     )
#     builder.add_edge("reporter", END)
#     return builder


# def build_graph_with_memory():
#     """Build and return the agent workflow graph with memory."""
#     # use persistent memory to save conversation history
#     # TODO: be compatible with SQLite / PostgreSQL
#     memory = MemorySaver()

#     # build state graph
#     builder = _build_base_graph()
#     return builder.compile(checkpointer=memory)


# def build_graph():
#     """Build and return the agent workflow graph without memory."""
#     # build state graph
#     builder = _build_base_graph()
#     return builder.compile()


# graph = build_graph()


import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
gemini_api = os.getenv("GOOGLE_API_KEY")
openrouter_api = os.getenv("OPENROUTER_API_KEY")
MAX_RECURSION_DEPTH=3

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    recursion_depth: int
    current_step: str

class CoordinatorNode:
    def __init__(self, llm_with_tools):
        self.llm_with_tools = llm_with_tools

    def __call__(self, state: AgentState):
        # logger.info(f"{Fore.RED}-------CoordinatorNode-------{Style.RESET_ALL}")
        # logger.info(f"state: {json.dumps(state_to_dict(state), indent=4, ensure_ascii=False)}")
        current_messages = state["messages"]
        response = self.llm_with_tools.invoke(current_messages)

        # Update the state with the new message
        update_state = {}
        
        update_state["messages"] = response
        update_state["current_step"] = "coordinator"

        return update_state
    
class ToolsNode:
    def __init__(self, tools_by_name):
        self.tools_by_name = tools_by_name

    def __call__(self, state: AgentState):
        # logger.info(f"{Fore.RED}-------ToolsNode-------{Style.RESET_ALL}")
        # logger.info(f"state: {json.dumps(state_to_dict(state), indent=4, ensure_ascii=False)}")
        current_messages = state["messages"]
        last_message = current_messages[-1]
        if not last_message.tool_calls:
            return state
        
        tool_messages_to_add = []
        tool_calls = last_message.tool_calls
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool_result = self.tools_by_name[tool_name].invoke(tool_args)
            tool_messages_to_add.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"], name=tool_name))

        updated_state = {}
        updated_state["messages"] = tool_messages_to_add
        updated_state["current_step"] = "tools"

        return updated_state
    
def routes_from_coordinator(state: AgentState):
    if state["recursion_depth"] >= MAX_RECURSION_DEPTH:
        return "end"
    else:
        if state["messages"][-1].tool_calls:
            state["recursion_depth"] += 1
            return "tools"
        else:
            return "end"
    
@tool
def web_search_tool(query: Annotated[str, "The search query to find information on the web"]) -> str:
    """
    Use this tool to search the web for current information when the vector store doesn't contain the answer.
    """
    try:
        search = TavilySearch(
                max_results=5,
                topic="general",
                include_answer="advanced", ## Include an LLM-generated answer to the provided query. basic or true returns a quick answer. advanced returns a more detailed answer.
                # include_raw_content=False,
                # include_images=False,
                # include_image_descriptions=False,
                # search_depth="basic",
                # time_range="day",
                # include_domains=None,
                # exclude_domains=None
            )
        
        results = search.invoke({"query": query})
        
        if not results:
            return f"No results found for: {query}"
        
    
        # Use this if you want direct search result from search index
        # response = ""
        # for i, result in enumerate(results, 1):
        #     response += f"{i}. {result['title']}\n"
        #     response += f"   Link: {result['url']}\n"
        #     response += f"   Snippet: {result['content']}\n\n"

        response = results["answer"] # Summary answer from LLM generated by Tavily
            
        return response
    except Exception as e:
        return f"Sorry, I couldn't perform the web search. Error: {str(e)}"
    
TOOLS = [web_search_tool]
TOOLS_BY_NAME = {
    "web_search_tool": web_search_tool,
}

def load_gpt(temperature=.0, model_name="gpt-4o", verbose=True, max_retries=3, timeout=None):
    llm = ChatOpenAI(
        # model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openrouter_api,  # if you prefer to pass api key in directly instaed of using env vars
        base_url="https://openrouter.ai/api/v1",
        # organization="...",
        # other params...
    )
    print(f'Successfully loaded {model_name} model')
    return llm

gpt_4o = load_gpt(model_name='gpt-4o')
gpt_4o_with_tools = gpt_4o.bind_tools(TOOLS)


coordinator_node = CoordinatorNode(gpt_4o_with_tools)
tool_node = ToolsNode(TOOLS_BY_NAME)


def _build_simple_qa_graph():
    builder = StateGraph(AgentState)
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("tools", tool_node)
    builder.add_edge("tools", "coordinator")
    builder.add_conditional_edges(
            "coordinator",
            routes_from_coordinator,
            {
                "tools": "tools",
                "end": END
            }
        )
    
    builder.set_entry_point("coordinator")
    return builder

def build_simple_qa_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_simple_qa_graph()
    return builder.compile()

simple_qa_graph = build_simple_qa_graph()

# Simple query execution for the simple QA graph
def run_simple_query(query: str):
    """Run a simple query on the simple QA graph."""
    result = simple_qa_graph.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "recursion_depth": 0,
            "current_step": "start"
        })
    return result

result = run_simple_query("How are you?")
print(result)
