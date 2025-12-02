import json
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage

from utils import read_text_file, tag_text, Agent

from typing_extensions import TypedDict, Union, List
from langchain.messages import HumanMessage, AIMessage


class ReceptionConversationState(TypedDict):
    conversation: List[Union[HumanMessage, AIMessage]]
    questionnaire: AIMessage
    answers: List[AIMessage]
    llm_calls: int
    answers_complete: bool


class ReceptionAgent(Agent):

    def __init__(self, max_llm_calls: int = 10, language: str = "german"):

        self.language = language
        self.max_llm_calls = max_llm_calls

        assert language in ["german"], "only german is supported at the moment"

        dictionary = read_text_file(f"./prompts/{language}/wörterbuch/algemein.md", "WÖRTERBUCH")
        self.answer_validation_system_message = SystemMessage("\n".join([
            *dictionary,
            *read_text_file(f'./prompts/{language}/system/frage_antwort_validierung.md', "SYSTEM")
        ]))

        self.question_answer_system_message = SystemMessage("\n".join([
            *dictionary,
            *read_text_file(f'./prompts/{language}/system/frage_antwort.md', "SYSTEM")
        ]))
        self.reception_system_message = SystemMessage("\n".join([
            *dictionary,
            *read_text_file(f'./prompts/{language}/system/empfang.md', "SYSTEM")
        ]))

        # Definiere LLM
        self.validation_llm = AzureChatOpenAI(
            api_version="2025-01-01-preview",
            azure_deployment="gpt-4.1",
            model="gpt-4.1",
            temperature=0.15,
            max_completion_tokens=1000,
            timeout=20,
            max_retries=1
        )
        self.reception_llm = AzureChatOpenAI(
            api_version="2025-01-01-preview",
            azure_deployment="gpt-4o",
            model="gpt4o",
            temperature=0.85,
            max_completion_tokens=1000,
            timeout=20,
            max_retries=1
        )

    def compile(self):

        agent_builder = StateGraph(ReceptionConversationState)

        agent_builder.add_node(self.human_input_node.__name__, self.human_input_node)
        agent_builder.add_node(self.receptionist_node.__name__, self.receptionist_node)
        agent_builder.add_node(self.question_answer_node.__name__, self.question_answer_node)
        agent_builder.add_node(self.answer_validation_node.__name__, self.answer_validation_node)

        agent_builder.add_edge(START, self.human_input_node.__name__)
        agent_builder.add_edge(self.human_input_node.__name__, self.receptionist_node.__name__)
        agent_builder.add_edge(self.receptionist_node.__name__, self.question_answer_node.__name__)
        agent_builder.add_edge(self.question_answer_node.__name__, self.answer_validation_node.__name__)
        agent_builder.add_conditional_edges(
            self.answer_validation_node.__name__,
            self.is_questionnaire_complete,
            [self.human_input_node.__name__, END],
        )

        return agent_builder.compile()
    
    def format_conversation(self, state: ReceptionConversationState):
        return [tag_text(str(c.content), "USER") if c.type == "human" else tag_text(str(c.content), "MITARBEITENDE") for c in state["conversation"]]

    def human_input_node(self, state: ReceptionConversationState):

        if len(state['conversation']) > 0 and state['conversation'][-1].type == "ai":
            print(f"REZEPTION   |{state['conversation'][-1].content}")

        # Get user input
        msg = input("INPUT       | ")
        
        return {
            "conversation": [*state["conversation"], HumanMessage(msg)]
        }


    def answer_validation_node(self, state: ReceptionConversationState):
        msg = self.validation_llm.invoke([
            *self.format_conversation(state),
            tag_text(str(state["questionnaire"].content), "FRAGEBOGEN"),
            tag_text(str(state["answers"][-1].content), "ANTWORTBOGEN") if len(state["answers"]) > 0 else ""
        ])

        if msg.content[-1] == "1":
            answer_complete = True
        else:
            answer_complete = False

        return {
            "answers_complete": answer_complete,
        }
    
    def question_answer_node(self, state: ReceptionConversationState):

        msg = self.validation_llm.invoke([
            self.question_answer_system_message,
            *self.format_conversation(state),
            tag_text(str(state["questionnaire"].content), "FRAGEBOGEN"),
        ])

        print(f"[DEBUG] - ANTWORTBOGEN:\n{msg.content}")

        return {
            "answers": [*state["answers"], msg],
        }

    def receptionist_node(self, state: ReceptionConversationState):

        # call reception with last 'answers'
        msg = self.reception_llm.invoke([
            self.reception_system_message,
            *self.format_conversation(state),
            tag_text(str(state["questionnaire"].content), "FRAGEBOGEN"),
            tag_text(str(state["answers"][-1].content), "ANTWORTBOGEN") if len(state["answers"]) > 0 else ""
        ])

        return {
            "conversation": [*state["conversation"], msg],
            "llm_calls": state["llm_calls"] + 1
        }

    def is_questionnaire_complete(self, state: ReceptionConversationState) -> str:

        print(f"[DEBUG] - ANSWERS COMPLETE: {state['answers_complete']}")
        
        if state["answers_complete"]:
            return END

        if state["llm_calls"] > self.max_llm_calls:
            return END

        return self.human_input_node.__name__
        