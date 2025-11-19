from os import environ
from dotenv import load_dotenv
from typing import Literal

from langchain_community.llms.vertexai import acompletion_with_retry
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import BaseTool, tool

# from langchain import agents 

from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
import pprint

from langgraph.graph import StateGraph, START, END


# @tool("document_tool", description="Führe eine Dokumentensuche aus und erhalte mit einer gegebenen search_query die top_k Dokumentenausschnitte.")
# def retreive_documents_tool(search_query: str, top_k: int):

#     azure_retreiver = AzureAISearchRetriever(
#         top_k = top_k,
#         api_key=environ["AZURE_AI_SEARCH_API_KEY"],
#         service_name=environ["AZURE_AI_SEARCH_SERVICE_NAME"],
#         index_name=environ["AZURE_AI_SEARCH_INDEX_NAME"],
#         content_key="chunk",
#     )

#     retrieved = azure_retreiver.invoke(search_query)
#     retrieved = [r.model_dump() for r in retrieved]

#     return [
#         { # TODO: inform downstram agents about this structure
#             "text_ausschnitt": r["chunk"], 
#             "ursprungs_dokument": r["title"]
#         }
#         for r in retrieved
#     ]

load_dotenv()

class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    scratchpad: Annotated[list[AIMessage], operator.add]
    llm_calls: int
    scratchpad_calls: int


reception_llm = AzureChatOpenAI(
    api_version="2025-01-01-preview",
    azure_deployment="gpt-4o",
    model="gpt4o",
    temperature=0,
    max_completion_tokens=1000,
    timeout=20,
    max_retries=1
)

scratchpad_system_message = SystemMessage(
    content=";".join([
        "you are given a scratchpad and a conversation",
        "do not change the information or the formatting on the scratchpad",
        "answer questions on the scratchpad by adding (and only adding) to the scratchpad",
        "keep the additions to the scratchpad very minimal and short, you dont want to fill it up with useless, irrelevant or redundand information"
    ])
)
scratchpad_validation_system_message = SystemMessage(content=";".join([
    "you are the validator of a scratchpad",
    "the scratchpad holds the initial scratchpad with open questions about the conversation",
    "these questions have been answered by an assistant",
    "check each answer of the scratchpad for having answered the initial question",
    "prefer prompt answers that are short and to the point",
    "do not allow for long winded answers",
    "return '1' if the scratchpad has been fully answered and filled correctly",
    "return '0' if the scratchpad has NOT been fully answered or answers are NOT clear"
]))

reception_system_message = SystemMessage(content=";".join([
    "you are a personal assistant that asks questions and collects information from an informal human request for information",
    "you have a scratchpad, structure your conversation around the scratchpads requirements.",

]))


# @tool(description="The scratchpad tool used for keeping track of vital questions and answers thruout a conversation")
# def scratchpad_tool(messages: list[str], scratchpad: str) -> str:
#
#     message = [
#         SystemMessage(
#             content=";".join([
#                 "you are given a scratchpad and a conversation",
#                 "do not change the information or the formatting on the scratchpad",
#                 "answer questions on the scratchpad by adding (and only adding) to the scratchpad",
#                 "keep the additions to the scratchpad very minimal and short, you dont want to fill it up with useless, irrelevant or redundand information"
#             ])
#         )
#     ] + messages + [scratchpad]
#
#     ret = scratchpad_llm.invoke(message)
#     con = ret.content
#
#     if type(con) == str:
#         return con
#     else:
#         raise NotImplementedError
# reception_llm = reception_llm.bind_tools([scratchpad_tool])

@tool(description="Validate a scratchpad and receive 0 for invalid scratchpads and 1 for valid scratchpads")
def scratchpad_validation(scratchpad: list[str]) -> bool:
    out = reception_llm.invoke([scratchpad_validation_system_message] + scratchpad)

    print(f"[DEBUG] tool output: {out}")

    if out.content == "1":
        return True
    else:
        return False

def scratchpad_call(state: dict):
    return {
        "scratchpad": [
            reception_llm.invoke([scratchpad_system_message] + [state["messages"][-1], state["scratchpad"][-1]]),
        ],
        "scratchpad_calls": state.get("scratchpad_calls", 0) + 1
    }

def scratchpad_validation_call(state: dict):

    print(f"[DEBUG] - validation call initiated with state {state['messages']}")

    out = []
    for tool_call in state["messages"][-1].tool_calls:

        print(f"[DEBUG] - called tool name: {tool_call['name']} with args: {tool_call['args']}")

        if tool_call["name"] == "scratchpad_validation":
            new_scratchpad = scratchpad_validation.invoke(tool_call["args"])
            scratchpad_validation.append(ToolMessage(new_scratchpad, tool_call_id=tool_call["id"]))

    return {
        "messages": out
    }

def reception_llm_call(state: dict):
    return {
        "messages": [
            reception_llm.invoke([reception_system_message] + state["messages"] + [state["scratchpad"][-1]])
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def is_scratchpad_valid(state: dict) -> Literal["reception_llm_call", END]:

    last_message = state["messages"][-1]

    print(f"[DEBUG] - last message: {type(last_message).__name__} - {last_message.content}")

    if last_message.tool_calls:

        if last_message.tool_calls[0]["messages"]:
            return END
        else:
            "reception_llm_call"


def main():

    agent_builder = StateGraph(ConversationState)

    agent_builder.add_node("reception_llm_call", reception_llm_call)
    agent_builder.add_node("scratchpad_call", scratchpad_call)
    agent_builder.add_node("scratchpad_validation_call", scratchpad_validation_call)

    agent_builder.add_edge(START, "reception_llm_call")

    agent_builder.add_edge("reception_llm_call", "scratchpad_call")
    agent_builder.add_edge("scratchpad_call", "scratchpad_validation_call")
    agent_builder.add_conditional_edges(
        "scratchpad_validation_call",
        is_scratchpad_valid,
        ["reception_llm_call", END],
    )


    agent = agent_builder.compile()

    messages = [HumanMessage(content="can you note down the birthday of 3 famous historical figures")]
    scratchpad = [AIMessage(content="1) What is the Topic of the question ?; 2) Is the question clearly formulated ?")]

    output = agent.invoke({
        "messages": messages,
        "scratchpad": scratchpad
    })

    pprint.pprint(output)

    # for m in messages["messages"]:
    #     print(f"[MESSAGES - {type(m).__name__}] - {m.content}")
    # print(f"[LLM_CALLS] - {messages['llm_calls']}")


if __name__ == "__main__":
    main()