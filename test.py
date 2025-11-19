from os import environ
from dotenv import load_dotenv
from typing import Literal, Sequence, Union

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

from langgraph.graph import MessagesState, StateGraph, START, END


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

def read_text_content(path: str):
    with open(path, "r") as f:
        return(f.read())



class ConversationState(TypedDict):
    konversation: Annotated[Sequence[Union[HumanMessage, AIMessage]], operator.add]
    fragebogen: AIMessage
    antwortbogen: AIMessage
    llm_calls: int


def create_reception_agent_system():
    
    # Definiere LLM
    llm = AzureChatOpenAI(
        api_version="2025-01-01-preview",
        azure_deployment="gpt-4o",
        model="gpt4o",
        temperature=0,
        max_completion_tokens=1000,
        timeout=20,
        max_retries=1
    )

    frage_antwort_system_message = SystemMessage(read_text_content("./prompts/german/system/frage_antwort.md"))
    frage_antwort_validation_system_message = SystemMessage(read_text_content("./prompts/german/system/frage_antwort_validierung.md"))
    reception_system_message = SystemMessage(read_text_content("./prompts/german/system/rezeptionist.md"))


    @tool(description="Validiert die Antworten auf dem Antwortbogen gegeben der Fragen auf dem Fragebogen")
    def frage_antwort_validierungs_tool(fragebogen: str, antwortbogen: str) -> str:
        msg = llm.invoke([
            frage_antwort_validation_system_message,
            AIMessage(fragebogen, type="fragebogen"),
            AIMessage(antwortbogen, type="antwortbogen")
        ])
        
        print(f"[DEBUG] - content from question answer validation {msg.content}")
        
        return str(msg.content) # TODO

    def frage_antwort_validierung_node(state: ConversationState):

        last_message =  state["konversation"][-1]

        assert last_message.type == "ai", "last message to make tool cal has to be ai"

        out = []
        for tool_call in last_message.tool_calls:
            ret = frage_antwort_validierungs_tool.invoke(tool_call["args"])
            return { "konversation": ret }
        

    def frage_antwort_node(state: ConversationState):
        msg = llm.invoke([
            frage_antwort_system_message,
            *state["konversation"],
            AIMessage(state["fragebogen"].content, type="fragebogen")
        ])
        return {
            "antwortbogen": msg.content,
        }

    def ist_antwortbogen_vollstaendig(validierungs_tool_output: str) -> Literal["rezeptionist_node", END]:
        
        print(f"[DEBUG] - Validation tool output into conditional edge: {validierungs_tool_output}")
        
        # assert last_message.type == "ai", "last message should be ai to make tool call, this is a logic error, if this appears at runtime, please revisit the code !"
        
        # if not last_message.tool_calls:
        #     return "rezeptionist_node"
        
        # TODO - This should only check the last character, not the entire message !
        # if last_message.tool_calls[0]:
        #     return END
        # else:
        #     "reception_llm_call"
        return END


    rezeptions_llm = llm.bind_tools([frage_antwort_validierungs_tool])

    def rezeptionist_node(state: ConversationState):
        msg = rezeptions_llm.invoke([
            reception_system_message,
            *state["konversation"],
            AIMessage(state["fragebogen"].content, type="fragebogen"),
            AIMessage(state["antwortbogen"].content, type="antwortbogen")
        ])

        return {
            "konversation": [*state["konversation"], msg],
            "llm_calls": state.get("llm_calls", 0) + 1
        }


    agent_builder = StateGraph(ConversationState)

    agent_builder.add_node(rezeptionist_node.__name__, rezeptionist_node)
    agent_builder.add_node(frage_antwort_node.__name__, frage_antwort_node)
    agent_builder.add_node(frage_antwort_validierung_node.__name__, frage_antwort_validierung_node)

    agent_builder.add_edge(START, rezeptionist_node.__name__)
    agent_builder.add_edge(rezeptionist_node.__name__, frage_antwort_node.__name__)
    agent_builder.add_edge(frage_antwort_node.__name__, frage_antwort_validierung_node.__name__)
    agent_builder.add_conditional_edges(
        frage_antwort_validierung_node.__name__,
        ist_antwortbogen_vollstaendig,
        [rezeptionist_node.__name__, END],
    )

    return agent_builder.compile()



# def scratchpad_validation_call(state: ConversationState):
#     current_scratchpad = state["scratchpad"]
#     last_message = state["conversation"][-1]
#     print(f"[DEBUG] - validation call with scratchpad: {current_scratchpad.type} - {current_scratchpad.content}")
#     out = []
#     for tool_call in last_message.tool_calls:
#         print(f"[DEBUG] - called tool name: {tool_call['name']} with args: {tool_call['args']}")
#         if tool_call["name"] == "scratchpad_validation":
#             new_scratchpad = scratchpad_validation.invoke(tool_call["args"])
#             out.append(ToolMessage(new_scratchpad, tool_call_id=tool_call["id"]))
#     return {
#         "messages": out
#     }



def main():

    # wir laden alle system variablen für unsere API anbindungen
    load_dotenv()

    agent_system = create_reception_agent_system()


    input_ = ConversationState(
        konversation=[HumanMessage("can you note down the birthday of 3 famous historical figures")],
        fragebogen = AIMessage("1) Was ist der Fachbereich der Frage(n) ?: \n2) Was sind die Frage(n) ?: "),
        antwortbogen = AIMessage(""),
        llm_calls=0,
    )

    out = agent_system.invoke(input_)

    


if __name__ == "__main__":
    main()