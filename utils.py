# from typing_extensions import TypedDict, Union, List
# from langchain.messages import HumanMessage, AIMessage
from typing_extensions import List

def read_text_file(path: str, tag: str) -> List[str]:
    with open(path, "r") as f:
        return [f"<{tag}>{a.strip().replace('\n', '')}</{tag}>" for a in f.readlines()]

def tag_text(text: str, tag) -> str:
    return f"<{tag}>{text}</{tag}>"

class Agent:
    def __init__(self):
        pass

    def compile(self):
        raise NotImplementedError
