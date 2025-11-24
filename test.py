from dotenv import load_dotenv
from langchain_core.messages import AIMessage
import pprint

from agents.reception import create_reception_agent
from utils import ReceptionConversationState

def main():

    # wir laden alle system variablen für unsere API anbindungen
    load_dotenv()

    reception_agent = create_reception_agent()

    input_ = ReceptionConversationState(
        conversation = [],
        questionnaire = AIMessage("0) Gibt es mehrere Fragen ?: \n1) Was ist der Fachbereich der Frage(n) ?: \n2) Was sind die Frage(n) ?: "),
        answers = AIMessage(""),
        llm_calls = 0,
        answers_complete = False
    )

    out = reception_agent.invoke(input_)


    print("----- KOVERSATION -----")
    for msg in out["konversation"]:
        pprint.pprint(f"{msg.type} - {msg.content}")

    print("----- ANTWORTBOGEN -----")
    pprint.pprint(out["antwortbogen"].content)

if __name__ == "__main__":
    main()