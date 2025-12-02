from os import environ
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage
from langchain_community.retrievers import AzureAISearchRetriever

from typing_extensions import TypedDict, Union, List, Literal
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool

from utils import read_text_file, tag_text


class DokumentElement(TypedDict):
    text: str
    quelle: str

class KonversationKlassifikation(TypedDict):
    ist_frage: Literal["ja", "nein"]
    frage_typ: Literal["technisch", "offen", "persoenlich"]
    frage_praezision: Literal["niedrig", "mittel", "hoch"]
    aktuelle_frage: str
    fachbereich: str

class GedankengangBewertung(TypedDict):
    bezug_auf_quellen: Literal["6", "5", "4", "3", "2", "1"] # hat der agent sich auf quellen bezogen ?
    bezug_auf_sachverhalt: Literal["6", "5", "4", "3", "2", "1"] # bezieght sich der agent auf den sachverhalt der frage ?
    gedankengang_effizienz: Literal["6", "5", "4", "3", "2", "1"] # haellt sich der agent kurz und zum punkt ?

class Gedankengang(TypedDict):
    frage: HumanMessage 
    gedanken: AIMessage
    antwort: AIMessage | None

    bewertung: GedankengangBewertung | None

class WissensAgentState(TypedDict):
    konversation: List[Union[HumanMessage, AIMessage]]
    klassifikation: KonversationKlassifikation | None
    
    llm_calls: int
    dokument_elemente_in_kontext: List[DokumentElement] | None
    korrektur: AIMessage | None

    # bewertung
    gedankengang: Gedankengang | None
    beispiele: List[Gedankengang]

# Konversation
# Gedankengang


def create_knowledge_agent(
    max_llm_calls: int = 10
):
    
    # Definiere LLM
    llm = AzureChatOpenAI(
        api_version="2025-01-01-preview",
        azure_deployment="gpt-4o",
        model="gpt4o",
        temperature=0.85,
        max_completion_tokens=1000,
        timeout=20,
        max_retries=1
    )

    azure_dokumentensuche = AzureAISearchRetriever(
        top_k = 1, # wird in der funktion selbst noch mal gesetzt
        api_key=environ["AZURE_AI_SEARCH_API_KEY"],
        service_name=environ["AZURE_AI_SEARCH_SERVICE_NAME"],
        index_name=environ["AZURE_AI_SEARCH_INDEX_NAME"],
        content_key="chunk",
    )
    
    def gedanken_node(state: WissensAgentState):
        
        user_frage = state["konversation"][-1]
        assert user_frage.type == "human", "letztes konversationselement muss von menschen formuliert worden sein"

        hat_korrektur = state["korrektur"] is not None
        hat_quellen = state["dokument_elemente_in_kontext"] is not None

        gedanken = llm.invoke([
            SystemMessage("Du denks ueber die Frage des Users nach und listest eien Gedankengang der zu einer plausieblen antwort fuehren kann, ohne die antwort selbst. beachte dabei (falls vorhanden) die Korrektur die dir von deinem Mitarbetier zugeteilt wird"), # TODO
            # state["korrektur"] if hat_korrektur else "", # TODO
            user_frage
        ])
            
        return {
            "gedankengang": {
                "frage": user_frage,
                "gedanken": gedanken
            }
        }

    def antwort_node(state: WissensAgentState):

        assert state["gedankengang"], "gedankengang sollte ausgefuellt sein"

        frage = state["gedankengang"]["frage"]
        gedanken = state["gedankengang"]["gedanken"]
        
        antwort = llm.invoke([
            SystemMessage("Du beantwortest die gegebene user Frage kurz in einem Paragraphen, und beziehe dich auf die Gedanken."), # TODO
            frage,
            gedanken,
        ])

        gedankengang = Gedankengang(
            frage = frage,
            gedanken = gedanken,
            antwort = antwort,
            bewertung = None
        )

        return {
            "konversation": [*state["konversation"], antwort],
            "llm_calls": state["llm_calls"] + 1,
            "gedankengang": gedankengang,

            # "korrektur": state["korrektur"],
            # "beispiele": state["beispiele"]
        }
    
    @tool(name_or_callable="Ein Werkzeug für die dokument-element-suche durch das mit hilfe von `search_query`, `top_k` Dokument")
    def dokument_suche_werkzeug(search_query: str, top_k: int) -> List[DokumentElement]:

        azure_dokumentensuche.top_k = max(0, min(top_k, 3))

        retrieved = azure_dokumentensuche.invoke(search_query)
        retrieved = [r.model_dump() for r in retrieved]

        return [DokumentElement(text = r["chunk"], quelle = r["title"]) for r in retrieved]

    def dokument_suche_werkzeug_node(state: WissensAgentState):

        letzte_nachricht = state["konversation"][-1]

        assert letzte_nachricht.type == "ai", "letzte Nachricht muss von KI sein, dies ist ein Logik problem"

        print(f"DEBUG: tool call: {letzte_nachricht.tool_calls[-1]["name"]}; echter name des tools: {dokument_suche_werkzeug.__repr_name__}")

        has_valid_tool_call = len(letzte_nachricht.tool_calls) > 0 and letzte_nachricht.tool_calls[-1]["name"] == dokument_suche_werkzeug.__repr_name__
        
        if has_valid_tool_call:
            ergebnisse = dokument_suche_werkzeug.invoke(letzte_nachricht.tool_calls[-1]["args"])
            # TODO format ergebnise

        return {
            "dokument_elemente_in_kontext": ergebnisse
        }

    def klassifiziere_konversation_node(state: WissensAgentState):
        return {
            "klassifikation": None # TODO
        }
    
    def bewertung_node(state: WissensAgentState):

        assert state["gedankengang"], "gedankengang muss ausgeefuellt werden bevor korrektur agent gecalled wird"

        bewertung = llm.invoke([
            # TODO: quellen informationen
            SystemMessage("Du bewertest anch schema ... "), # TODO
            state["gedankengang"]["frage"],
            state["gedankengang"]["gedanken"],
            state["gedankengang"]["antwort"],
        ])

        return {
            "gedankengang": {
                "bewertung": bewertung
            }
        }

    def korrektur_node(state: WissensAgentState):

        assert state["gedankengang"], "gedankengang muss ausgeefuellt werden bevor korrektur agent gecalled wird"
        assert state["gedankengang"]["bewertung"], "bewertung muss ausgefuellt werden bevor korrektur agent gecalled wird"

        korrektur = llm.invoke([
            # TODO: quellen informationen
            SystemMessage("Du kommunizierst eine Korrektur zu dem gedanken agent, gibt dem gedanken agent alle informationen ueber welche teile der bewertung nicht gut ausgefallen sind und wo die fehler in dem gedankengang zu finden sind."), # TODO
            state["gedankengang"]["frage"],
            state["gedankengang"]["gedanken"],
            state["gedankengang"]["antwort"],
            # state["gedankengang"]["bewertung"] # TODO
        ])

        return {
            "korrektur": korrektur
        }
    
    def ist_die_bewertung_ausreichend(state: WissensAgentState):
        return END # else return korrektur_node.__name__
        

    agent_builder = StateGraph(WissensAgentState)
    agent_builder.add_node(antwort_node.__name__, antwort_node)
    agent_builder.add_node(gedanken_node.__name__, gedanken_node)
    agent_builder.add_node(klassifiziere_konversation_node.__name__, klassifiziere_konversation_node)
    agent_builder.add_node(dokument_suche_werkzeug_node.__name__, dokument_suche_werkzeug_node)
    agent_builder.add_node(korrektur_node.__name__, korrektur_node)
    agent_builder.add_node(bewertung_node.__name__, bewertung_node)
    
    agent_builder.add_edge(START, klassifiziere_konversation_node.__name__)
    agent_builder.add_edge(klassifiziere_konversation_node.__name__, dokument_suche_werkzeug_node.__name__) # TODO : entweder kriegt das klassifiziere_konversation_node den tool call, oder der gedanke node der noch einmal gecalled wird danach ...
    agent_builder.add_edge(dokument_suche_werkzeug_node.__name__, gedanken_node.__name__)
    agent_builder.add_edge(gedanken_node.__name__, antwort_node.__name__) # TODO : vieleicht fuegen wir den bewertungs node und den feedback node nach dem gedanke node, vor den antwort node ?
    # agent_builder.add_edge(gedanken_node.__name__, bewertung_node.__name__)
    # agent_builder.add_edge(bewertung_node.__name__, korrektur_node.__name__)
    
    agent_builder.add_edge(antwort_node.__name__, bewertung_node.__name__)
    agent_builder.add_conditional_edges(
        bewertung_node.__name__,
        ist_die_bewertung_ausreichend,
        [korrektur_node.__name__, END],
    )
    agent_builder.add_edge(korrektur_node.__name__, gedanken_node.__name__)

    # TODO: problem jetzt grade ist das wen wir dem gedanken_node das suchwerkzeug geben 
    # wird er erst nach dem reasoning sich auch die quellen beziehen koennen, 
    # aber es sollte vorher geschehen

    return agent_builder.compile()

