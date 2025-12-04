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
    hat_frage: Literal["ja", "nein"]
    frage_typ: Literal["technisch", "offen", "persoenlich"] | None
    frage_praezision: Literal["niedrig", "mittel", "hoch"] | None
    aktuelle_frage: str | None
    fachbereich: str | None

class GedankengangBewertung(TypedDict):
    bezug_auf_quellen: Literal["6", "5", "4", "3", "2", "1"] # hat der agent sich auf quellen bezogen ?
    bezug_auf_sachverhalt: Literal["6", "5", "4", "3", "2", "1"] # bezieght sich der agent auf den sachverhalt der frage ?
    gedankengang_effizienz: Literal["6", "5", "4", "3", "2", "1"] # haellt sich der agent kurz und zum punkt ?
    korrektur: str

class Gedankengang(TypedDict):
    frage: AIMessage 
    gedanken: AIMessage
    antwort: AIMessage
    bewertung: GedankengangBewertung

class WissensAgentState(TypedDict):
    konversation: List[Union[HumanMessage, AIMessage]]
    klassifikation: KonversationKlassifikation | None
    
    llm_calls: int
    dokument_elemente_in_kontext: List[DokumentElement] | None

    # bewertung
    gedankengang: Gedankengang | None
    beispiele: List[Gedankengang]


@tool(name_or_callable="dokument_suche_werkzeug", description="Ein Werkzeug für die dokument-element-suche durch das mit hilfe von `suchanfrage`, `top_k` Dokument")
def dokument_suche_werkzeug(suchanfrage: str, top_k: int) -> List[DokumentElement]:

    top_k = max(0, min(top_k, 10))

    azure_dokumentensuche = AzureAISearchRetriever(
        top_k = top_k,
        api_key=environ["AZURE_AI_SEARCH_API_KEY"],
        service_name=environ["AZURE_AI_SEARCH_SERVICE_NAME"],
        index_name=environ["AZURE_AI_SEARCH_INDEX_NAME"],
        content_key="chunk"
    )

    retrieved = azure_dokumentensuche.invoke(suchanfrage)

    retrieved = [r.model_dump() for r in retrieved]

    for r in retrieved:
        assert "page_content" in r, "'page_content' ist nicht der kontent key des vektorsuchen-outputs, bitte den verwendeten key der Vektorsuche fuer text kontent angeben"
        assert "title" in r["metadata"], "'title' ist nicht der quellen key des vektorsuchen-outputs, bitte den verwendeten key der Vektorsuche fuer text quelle angeben"

    return [DokumentElement(text = r["page_content"], quelle = r["metadata"]["title"]) for r in retrieved]


class WissensAgent:

    def __init__(self,
        max_llm_calls: int = 10,
        erwuenschte_note: float = 3.0
    ):
        self.max_llm_calls = max_llm_calls
        self.erwuenschte_note = erwuenschte_note
    
        self.llm = AzureChatOpenAI(
            api_version="2025-01-01-preview",
            azure_deployment="gpt-4o",
            model="gpt4o",
            temperature=0.65,
            max_completion_tokens=1000,
            timeout=20,
            max_retries=1
        )

    def compile(self):
        workflow = StateGraph(WissensAgentState)
        workflow.add_node(self.user_input_node.__name__,              self.user_input_node)
        workflow.add_node(self.konversation_node.__name__,            self.konversation_node)
        workflow.add_node(self.dokument_suche_werkzeug_node.__name__, self.dokument_suche_werkzeug_node)
        workflow.add_node(self.gedanken_node.__name__,                self.gedanken_node)
        
        workflow.add_edge(START, self.user_input_node.__name__)
        workflow.add_conditional_edges(
            self.user_input_node.__name__, 
            self.konversation_hat_frage,
            {
                "ja": self.dokument_suche_werkzeug_node.__name__, 
                "nein": self.konversation_node.__name__
            }
        )
        workflow.add_edge(self.konversation_node.__name__, self.user_input_node.__name__)
        
        workflow.add_edge(self.dokument_suche_werkzeug_node.__name__, self.gedanken_node.__name__)
        # workflow.add_edge(self.gedanken_node.__name__, END)
        workflow.add_conditional_edges(
            self.gedanken_node.__name__,
            self.ist_gedankengang_zu_ende,
            {
                True: END, 
                False: self.dokument_suche_werkzeug_node.__name__, 
            }
        )

        # workflow.add_edge(gedanken_node.__name__, user_input_node.__name__)
        # TODO : vieleicht fuegen wir den bewertungs node und den feedback node nach dem gedanke node, vor den antwort node ?

        return workflow.compile()
    
    def user_input_node(self, state: WissensAgentState):

        print("----------------------------------------------------")
        
        if len(state['konversation']) > 0 and state['konversation'][-1].type == "ai":
            print(f"AGENT: {state['konversation'][-1].content}")

        if len(state['konversation']) > 0 and state['konversation'][-1].type == "human":
            konversation = state["konversation"]
        else:
            user_input = input("INPUT: ")
            konversation = [*state["konversation"], HumanMessage(user_input)]

        frage_klassifikations_llm = self.llm.with_structured_output(KonversationKlassifikation)
        klassifikation = frage_klassifikations_llm.invoke(f"""
            Klassifiziere die folgende konversation:
            ---
            KONVERSATION:
            {konversation}
            ---
            Achte auf die letzten Elemente der konversation und Klassifiziere die letzten User elemente wie folgt:
            beinhaltet die konversation eine user Frage ?
            wen Frage vorhanden, welche Frage typ passt am besten: "technisch", "offen", "persoenlich" ?
            wen Frage vorhanden, wie genau (speziell) ist die Frage: "niedrig", "mittel", "hoch" ?
            wen Frage vorhanden, was ist die aktuelle Frage ?
            wen Frage vorhanden, was ist der fachbereich der Frage ?
        """)

        print(f"[DEBUG] - frage klassifikation: {klassifikation}")

        return {
            "konversation": konversation,
            "klassifikation": klassifikation
        }
    

    def konversation_hat_frage(self, state: WissensAgentState):
        assert state["klassifikation"], "die konversation muss klassifiziert worden sein"
        print(f"[DEBUG] - Konversation hat frage: {state['klassifikation']['hat_frage']}")
        return state["klassifikation"]["hat_frage"]


    def konversation_node(self, state: WissensAgentState):

        print(f"[DEBUG] - antworte normal da keine frage vorhanden scheint.")

        antwort = self.llm.invoke([
            SystemMessage("""
                Du fuehrst eine Konversation mti einem User.
                Das Ziehl der KOnversation ist es eine frage zu beantworten.
                Der User sollte eine Frage stellen, bitte ihn/sie freundlicherweise darum.
            """),
            *state['konversation']
        ])

        return {
            "konversation": [*state['konversation'], antwort],
            "llm_calls": state["llm_calls"] + 1
        }


    def dokument_suche_werkzeug_node(self, state: WissensAgentState):
        
        assert state['klassifikation'], "klassifikation muss geschehen sein bevor das hier gecalled wird"

        user_frage = state['klassifikation']["aktuelle_frage"]
        user_frage_infos = {
            "fachbereich": state['klassifikation']['fachbereich'],
            "frage_praezision": state['klassifikation']['frage_praezision'],
            "frage_typ": state['klassifikation']["frage_typ"]
        }

        dokument_suche_llm = self.llm.bind_tools([dokument_suche_werkzeug])
        werkzeug_ausgabe = dokument_suche_llm.invoke(f"""
            Du kriegst eine User Frage sowie informationionen ueber diese Frage.
            
            USER_FRAGE: {user_frage}
            Information ueber USER_FRAGE: {user_frage_infos}
                                                     
            Du hast ein Werkzeug 'dokument_suche_werkzeug'.
            Mittels dieses Werkzeuges formuliere eine 'suchanfrage' um 'top_k' Dokumentauszuege zu finden die fuer die beantwortung der USER_FRAGE relevant sind.
            Die 'suchanfrage' sollte die USER_FRAGE beschreiben, die Informationen mit beinhalten, und fuer eine Vektor suche ausgelegt sein.
            Die 'suchanfrage' wird via Vektorsuche an ein LLM gegeben um relevante Dokumentauszuege zu finen, beachte die formatierung die solche LLM-encoder erwarten.
            Diese Dokumentauszuege sind relevant zur beantwortung der USER_FRAGE.
            Benutze hierbei 'top_k' um bei praeziesen Fragen weniger Dokumentauszuege zu kriegen, und bei offeneren Fragen mehr Dokumentauszuege zu kriegen.
            'top_k' ist auf einer skala von 1 bis 10, wo 1 = ein Dokumentauszug ist und 10 = viele Dokumentenauszuege.
        """)

        has_valid_tool_call = len(werkzeug_ausgabe.tool_calls) > 0 and werkzeug_ausgabe.tool_calls[-1]["name"] == 'dokument_suche_werkzeug'
        if has_valid_tool_call:
            print(f"[DEBUG] - werkzeugaufruf mit {werkzeug_ausgabe.tool_calls[-1]['args']}")
            ergebnisse = dokument_suche_werkzeug.invoke(werkzeug_ausgabe.tool_calls[-1]["args"])

            # TODO format ergebnise

            return {
                "dokument_elemente_in_kontext": ergebnisse
            }
        return {}
    
    def gedanken_node(self, state: WissensAgentState):

        assert state["klassifikation"], "klassifikation muss stadtgefunden haben zu diesem zeitpunkt"

        bewertung_llm = self.llm.with_structured_output(GedankengangBewertung)

        user_frage = state['klassifikation']["aktuelle_frage"]

        gedanken = self.llm.invoke(f"""
            Du kriegst:
            - Eine user Frage
            - Informationen ueber diese Frage
            - eine liste an dokumentausschnitte und deren quellen
            - fals vorhanden eine Korrektur ueber deine vorherigen versuche.

            USER_FRAGE: {user_frage}
            Information ueber USER_FRAGE: {state['klassifikation']}
            DOKUEMNTEAUSSCHNITTE und QUELLEN: {state['dokument_elemente_in_kontext']}
            KORREKTUR: {state['gedankengang']['bewertung']['korrektur'] if state['gedankengang'] else "KEINE"}.

            Du denks ueber die Frage des Users nach.
            Du versuchst durch die Korrektur vorherige fehler zu vermeiden.
            Du beziehst dich zu jedem moeglichen zeitpunkt auf die quellen und dokumentausscnitte da sie die besten Informationen enthalten.
            
            Liste dein Gedankengang auf in welchem du alle relevanten informationen durchlaeufst.
            Beantworte die Frage NICHT sondern beschreibe nur den Gedankengang.
        """)
        
        antwort = self.llm.invoke(f"""
            Du kriegst:
            - Eine User Frage
            - Ein Gedankengang in welchem die rueckfuehrung der Logik die die Frage beantworten soll beschrieben ist.
            - eine liste an Dokumentausschnitten und quellen.
            
            USER_FRAGE: {user_frage}
            DOKUEMNTEAUSSCHNITTE und QUELLEN: {state['dokument_elemente_in_kontext']}
            GEDANKENGANG: {gedanken.content} 
            
            Du beantwortest die gegebene USER_FRAGE anhand der DOKUMENTAUSSCHNITTE, deren QUELLEN und den GEDANKENGANG.
            Antworte kurz, in einem Paragraphen.
            Verwende in deiner Antwort die quellenangaben aus DOKUMENTENAUSSCHNITTE und QUELLEN die du verwendest mit der notation [QUELLE].
        """)

        string_formatierte_beispiele = '\n'.join([
            f"""
                'frage': {beispiel['frage']}
                'gedanken': {beispiel['gedanken']}
                'bezug_auf_quellen': {beispiel['bewertung']['bezug_auf_quellen']}
                'bezug_auf_sachverhalt': {beispiel['bewertung']['bezug_auf_sachverhalt']}
                'gedankengang_effizienz': {beispiel['bewertung']['gedankengang_effizienz']}
            """
            for beispiel in state['beispiele']
        ])

        bewertung = bewertung_llm.invoke(f"""
            Du kriegst:
            - Eine Liste an Beispielen welche du als Wichtigste referenz siehst.
            - Eine User Frage
            - Ein Gedankengang in welchem die rueckfuehrung der Logik die die Frage beantworten soll beschrieben ist.
            - eine liste an Dokumentausschnitten und quellen.
            - Die Antwort auf die User Frage

            ---
            BEISPIElE:
                                         
            {string_formatierte_beispiele}
            
            ---

            USER_FRAGE: {user_frage}
            DOKUEMNTEAUSSCHNITTE und QUELLEN: {state['dokument_elemente_in_kontext']}
            GEDANKENGANG: {gedanken.content} 
            ANTWORT: {antwort.content}

            ---

            Bewerte den Gedankengang anhand des Deutschen Schulnoten Systems fuer die Faktoren.
            "1" bedeutet sehr gut und "6" bedeutet sehr schlecht, alles dazwischen ist eine jehweilige Einstufung der Bewertung.
            Bewerte den GEDANKENGANG nach diesem System auf den die Folgenden Faktoren:
            - bezug_auf_quellen: wie gut bezieht sich der gedankengang auf die vorhandenen quellen, und werden diese sinnvoll genutzt um eine logik zu erklaeren.
            - bezug_auf_sachverhalt: wie gut wird sich auf den sachverhalt der frage bezogen waehrend des gedankengangs.
            - gedankengang_effizienz - wie effizient die gedanken verwendet werden, ob sachen ausgefuehrt werden die irrelevant sind und wie schnell sich auf eine konkrete logik bezogen wird.
            Fuege eine korrektur hinzu in welcher du darauf hinweist wieso du gute oder schlechte noten verteilt hast, diese sollen so formuliert werden das der GEDANKENGANG genau weis was die fehler / probleme waren.
            Bewerte den GEDANKENGANG anhand der BEISPIELE, diese Beispiele haben eine Perfekte Notenvergabe welche du emulieren solltest. 
        """)

        print(f"[DEBUG] - bewertung: {bewertung}")

        return {
            "konversation": [*state["konversation"], antwort],
            "llm_calls": state["llm_calls"] + 1,
            "gedankengang": {
                "frage": AIMessage(user_frage),
                "gedanken": gedanken,
                "antwort": antwort,
                "bewertung": bewertung
            }
        }
    
    def ist_gedankengang_zu_ende(self, state: WissensAgentState):
                
        assert state['gedankengang'], "gedankengang sollte gesetzt sein an diesem punkt"

        note = sum([
            int(state['gedankengang']['bewertung']["bezug_auf_quellen"]),
            int(state['gedankengang']['bewertung']["bezug_auf_sachverhalt"]),
            int(state['gedankengang']['bewertung']['gedankengang_effizienz'])
        ]) / 3

        print(f"[DEBUG] - note ereicht: {note}")

        if note <= self.erwuenschte_note:
            return True

        if state['llm_calls'] >= self.max_llm_calls:
            print("[DEBUG] - maximale anzahl an llm calls ereicht")
            return True
        
        return False
        
