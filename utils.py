from typing_extensions import TypedDict, Union, List
from langchain.messages import HumanMessage, AIMessage

def read_text_file(path: str, tag: str) -> str:
    with open(path, "r") as f:
        s = f.read()
        s = f"<{tag}>{s}</{tag}>"
    return s

def tag_text(text: str, tag) -> str:
    return f"<{tag}>{text}</{tag}>"


class ReceptionConversationState(TypedDict):
    conversation: List[Union[HumanMessage, AIMessage]]
    questionnaire: AIMessage
    answers: AIMessage
    llm_calls: int
    answers_complete: bool
