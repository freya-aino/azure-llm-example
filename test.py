import polars as pl
import json
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
import pprint

def main():

    # wir laden alle system variablen für unsere API anbindungen
    load_dotenv()

    # from agents.reception import ReceptionAgent
    # reception_agent = ReceptionAgent(max_llm_calls=10, language="german")
    # agent = reception_agent.compile()


    # input_ = ReceptionConversationState(
    #     conversation = [],
    #     questionnaire = AIMessage("0) Gibt es mehrere Fragen ?: \n1) Was ist der Fachbereich der Frage(n) ?: \n2) Was sind die Frage(n) ?: "),
    #     answers = [],
    #     llm_calls = 0,
    #     answers_complete = False
    # )

    # out = agent.invoke(input_)

    # print("----- KOVERSATION -----")
    # for msg in out["conversation"]:
    #     pprint.pprint(f"{msg.type} - {msg.content}")

    # print("----- ANTWORTBOGEN -----")
    # pprint.pprint(out["questionnaire"].content)

    df = pl.read_csv("./beispiele.csv")

    print(df)

    from agents.knowledge import WissensAgent, WissensAgentState
    
    output = WissensAgent().compile().invoke(
        WissensAgentState(
            konversation=[],
            klassifikation=None,
            llm_calls=0,
            dokument_elemente_in_kontext=[],
            gedankengang=None,
            beispiele=[] # TODO
        )
    )

    pprint.pprint(output)

    # Step 1 - User Step : Get Answeres to fill Questionair.
    # Step 2 - LLM  Step : Compile Questionair.
    # Step 3 - 

    # Agent 1 - Konversationsagent
    # Agent 2 - Wissensagent
    # Agent 3 - Bewertungsagent

    # [anhand Menschlicher Beispiele]
    # [Bewertung des reasoning einmal definieren]
    # [syntetische Reasoining generierung mit: beispiels-fragen (8) -> Wissensagent (hoere temperatur, etc.) -> 5 reasonings pro Beipiel (40)]


if __name__ == "__main__":
    main()