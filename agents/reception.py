from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage

from utils import ReceptionConversationState, read_text_file, tag_text

def create_reception_agent():
    
    # Definiere LLM
    reception_llm = AzureChatOpenAI(
        api_version="2025-01-01-preview",
        azure_deployment="gpt-4o",
        model="gpt4o",
        temperature=0.85,
        max_completion_tokens=1000,
        timeout=20,
        max_retries=1
    )
    validation_llm = AzureChatOpenAI(
        api_version="2025-01-01-preview",
        azure_deployment="gpt-4o",
        model="gpt4o",
        temperature=0.25,
        max_completion_tokens=1000,
        timeout=20,
        max_retries=1
    )

    frage_antwort_system_message = read_text_file('./prompts/german/system/frage_antwort.md', "system")
    frage_antwort_validation_system_message = read_text_file('./prompts/german/system/frage_antwort_validierung.md', "system")
    reception_system_message = read_text_file('./prompts/german/system/rezeptionist.md', "system")


    def frage_antwort_validierung_node(state: ReceptionConversationState):

        msg = validation_llm.invoke([
            frage_antwort_validation_system_message,
            tag_text(str(state["questionnaire"].content), "fragebogen"),
            tag_text(str(state["answers"].content), "antwortbogen")
        ])

        is_antwortbogen_valid = msg.content[-1]
        if is_antwortbogen_valid == "1":
            is_antwortbogen_valid = True
        else:
            is_antwortbogen_valid = False

        return {
            "antwortbogen_vollstaendig": is_antwortbogen_valid
        }
        

    # llm_mit_tool = llm.bind_tools([frage_antwort_validierungs_tool])

    def frage_antwort_node(state: ReceptionConversationState):

        msg = validation_llm.invoke([
            frage_antwort_system_message,
            *state["conversation"], 
            tag_text(str(state["questionnaire"].content), "fragebogen"),
            tag_text(str(state["answers"].content), "antwortbogen")
        ])

        return {
            "answers": msg,
        }


    def rezeptionist_node(state: ReceptionConversationState):

        if len(state['conversation']) > 0 and state['conversation'][-1].type == "ai":
            print(f"REZEPTION   |{state['conversation'][-1].content}")

        # Get user input
        msg = input("INPUT       |")
        
        state["conversation"].append(HumanMessage(msg))

        msg = reception_llm.invoke([
            reception_system_message,
            *state["conversation"],
            tag_text(str(state["questionnaire"].content), "fragebogen"),
            tag_text(str(state["answers"].content), "antwortbogen")
        ])

        return {
            "conversation": [*state["conversation"], msg],
            "llm_calls": state["llm_calls"] + 1
        }

    def ist_antwortbogen_vollstaendig(state: ReceptionConversationState) -> str:
        
        if state["answers_complete"]:
            return END

        if state["llm_calls"] > 10:
            return END

        return rezeptionist_node.__name__
        

    agent_builder = StateGraph(ReceptionConversationState)

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

