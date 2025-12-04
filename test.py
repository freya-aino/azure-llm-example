import polars as pl
from dotenv import load_dotenv
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


    from agents.knowledge import WissensAgent, WissensAgentState, Gedankengang, GedankengangBewertung

    df = pl.read_excel("./beispiele.xlsx")
    beispiele = [
        {
            "frage": r['orginal_frage'],
            "gedanken": r['gedanken'],
            "antwort": r['antwort'],
            "bewertung": {
                "bezug_auf_quellen": str(r['bewertung_bezug_auf_quellen']),
                "bezug_auf_sachverhalt": str(r['bewertung_bezug_auf_sachverhalt']),
                "gedankengang_effizienz": str(r['bewertung_gedankengang_effizienz']),
            }
        }
        for r in df.iter_rows(named=True)
    ]
    
    outputs = []
    
    
    agent = WissensAgent(
        max_llm_calls = 3,
        erwuenschte_note = 2.4
    ).compile()

    
    print(agent.get_graph().draw_mermaid())
    exit()


    o = agent.invoke(
        WissensAgentState(
            konversation=[],
            klassifikation=None,
            llm_calls=0,
            dokument_elemente_in_kontext=[],
            gedankengang=None,
            beispiele=beispiele
        )
    )

    print(f"[FINALE ANTWORT]: {o['gedankengang']['antwort'].content}")

    

if __name__ == "__main__":
    main()