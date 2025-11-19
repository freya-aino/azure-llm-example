from os import environ
from dotenv import load_dotenv
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI

from langchain import agents #  import AzureChatOpenAI, AzureOpenAI
from langchain.tools import BaseTool, tool



USER_FRAGE_ELEMENTE_NOTATION = "<user_frage_element>", "</user_frage_element>"
WOERTERBUCH_NOTATION = "<wörterbuch>", "</wörterbuch>"
GLOBALE_SYSTEMPROMPTS_NOTATION = "<system>", "</system>"
AGENT_PROMPT_NOTATION = "<agent>", "</agent>"
DOKUMENT_RAG_TOOL_NAME = "retreive_document_info"

# TODO: is this usefull ?
# @tool("scratchpad")
# def scratchpad()

@tool(name_or_callable=DOKUMENT_RAG_TOOL_NAME)
def retreive_documents_tool(search_query: str, top_k: int):

    azure_retreiver = AzureAISearchRetriever(
        top_k = top_k,
        api_key=environ["AZURE_AI_SEARCH_API_KEY"],
        service_name=environ["AZURE_AI_SEARCH_SERVICE_NAME"],
        index_name=environ["AZURE_AI_SEARCH_INDEX_NAME"],
        content_key="chunk",
    )

    retrieved = azure_retreiver.invoke(search_query)
    retrieved = [r.model_dump() for r in retrieved]

    return [
        { # TODO: inform downstram agents about this structure
            "text_ausschnitt": r["chunk"], 
            "ursprungs_dokument": r["title"]
        }
        for r in retrieved
    ]


WOERTERBUCH = [
    "AUFGABE=Deine Aufgabe, welche dir zugeteilt wird, welche du Auschließlich behandelst und von welcher du unter keinen Umständen abweichst.",
    "USERFRAGE=Die ursprüngliche Frage des Users welche du nicht direkt erhälst",
    f"USERFRAGE_ELEMENTE=Die Elemente der User Frage welche für dich Relevant sind und Vorverarbeitet wurden, sie werden mit {USER_FRAGE_ELEMENTE_NOTATION} gekenzeichnet und sind die Wichtigsten Informationen für deine AUFGABE.",
    "WISSENSDATENBANK=Eine Datenbank welche informationen über alle notwendigen Themen enthällt mit welcher du deine AUFGABE bewältigst.",
    "DOKUMENT=Ein Dokument das in der WISSENSDATENBANK Enthalten ist, dessen Informationen als 100% Wahr und Genau behandelt werden.",
    "TEXTSEQUENZ=Ein Text welcher direkt aus einem DOKUMENT stammt, diese Informationen sind mit höchster präferenz für richtigkeit und genaugkeit zu werten!",
    "QUELLE=Eine Referenz designiert mit Wissenschaftlicher notation ([1], [2], etc.) sowie eine Zuordnung der Referenz zu einer TEXTSEQUENZ und DOKUMENT.",
    f"FRAGETOOL=Ein Tool namens '{DOKUMENT_RAG_TOOL_NAME}' mit welchem du mittels der parameter 'search_query' und 'top_k' genau 'k' (anzahl) TEXTSEQUENZ aus der WISSENSDATENBANK, anhand der frage in 'search_query', extrahieren kannst.",
    "TEAM=Wichtige Mitarbeitende welche einzelne speziealisierte Aufgaben erledigen auf wessen informationen du entweder zurückgreifen kannst oder welche du kontolieren musst um Effizient deine spezielle AUFGABE zu erfüllen."
]

GLOBALE_SYSTEMPROMPTS = [
    "Wörter die nur mit großbustaben geschrieben werden beschreiben immer die selbe entität (z.b. eine Datenstruktur, ein bestimmter Text)",
    "Du erhälst ein Wörterbuch mit einer liste an Bezeichnungen die Global, in jeder situation gültig sind.",
    "Du bist Teil eines Teams welches Dokumente verarbeitet",
    "Eure TEAM AUFGABE ist es zusammen eine akurate und präzise Auskunft zu Fragen zu formulieren die in den zu verwaltenen Dokumenten nach Antworten zu suchen.",
    "Jeder von euch in TEAM erhält eine eigene AUFGABE",
    "Die persönliche AUFGABE hat primäre präferenz.",
    "Die TEAM AUFGABE hat sekundäre preferenz.",
]

WISSENSAGENT_REASONING_PROMPTS = [
    "AUFGABE: Du erhälst USERFRAGE_ELEMENTE und benutze FRAGETOOL um beliebig viele (TEXTSEQUENZ, QUELLE) Paare zu Erhalten welche relevant for USERFRAGE_ELEMENTE sind.",
    "Verwende auschließlich Informationen vom FRAGETOOL",
    "Beantworte die USERFRAGE_ELEMENTE nicht direkt sondern schreibe deinen Gedankengang in welchem du eine Logische rückführung ermittelst wie und wo die TEXTSEQUENZ und QUELLE informationen die elemente der USERFRAGE_ELEMENTE beantworten.",
    "Schreibe diesen gedankengang in 3-4 Paragraphen mit dem titel 'Gedanken'",
    "Verwende QUELLE für jeden Bezug auf TEXTSEQUENZ.",
]

WISSENSAGENT_ANTWORT_PROMPTS = [
    "AUFGABE: Du erhälst einen Gedankengang, gekenzeichnet mit 'Gedanken' und eine liste an quellen gekenzeichnet mit 'Quellen' der auf der basis von TEXTSEQUENZ produziert wurde, und beantworte die USERFRAGE_ELEMENTE.",
    "Beziehe dich in deiner Antwort ausschließlich auf die Logik die in den Gedanken erläutert wurden, sowie die informationen aus der WISSENSDATENBANK und TEXTSEQUENZ",    "Formuliere deine Antwort in 1-2 Paragraphen",
    "Beziehe dich auf QUELLE für jedes TEXTSEQUENZ die erwähnt wurde.",
]

WOERTERBUCH_FORMATIERT = [f"{WOERTERBUCH_NOTATION[0]}{p}{WOERTERBUCH_NOTATION[1]}" for p in WOERTERBUCH]
GLOBALE_SYSTEMPROMPTS_FORMATIERT = [f"{GLOBALE_SYSTEMPROMPTS_NOTATION[0]}{p}{GLOBALE_SYSTEMPROMPTS_NOTATION[1]}" for p in GLOBALE_SYSTEMPROMPTS]
WISSENSAGENT_REASONING_PROMPTS_FORMATIERT = [f"{AGENT_PROMPT_NOTATION[0]}{p}{AGENT_PROMPT_NOTATION[1]}" for p in WISSENSAGENT_REASONING_PROMPTS]
WISSENSAGENT_ANTWORT_PROMPTS_FORMATIERT = [f"{AGENT_PROMPT_NOTATION[0]}{p}{AGENT_PROMPT_NOTATION[1]}" for p in WISSENSAGENT_ANTWORT_PROMPTS]


def main():
    load_dotenv()

    llm_gpt_4o = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-11-20",
        temperature=0.1,
        timeout=30,
    )

    agent_01 = agents.create_agent(
        model=llm_gpt_4o,
        # tools=[scratchpad],
        system_prompt="\n".join([]) # TODO
    )

    agent_02_1 = agents.create_agent(
        model=llm_gpt_4o,
        tools=[retreive_documents_tool],
        system_prompt = "\n".join([*WOERTERBUCH_FORMATIERT, *GLOBALE_SYSTEMPROMPTS_FORMATIERT, *WISSENSAGENT_REASONING_PROMPTS_FORMATIERT])
    )
    agent_02_2 = agents.create_agent(
        model=llm_gpt_4o,
        system_prompt="\n".join([*WOERTERBUCH_FORMATIERT, *GLOBALE_SYSTEMPROMPTS_FORMATIERT, *WISSENSAGENT_ANTWORT_PROMPTS_FORMATIERT])
    )

    # agent_03 = agents.create_agent() # TODO


if __name__ == "__main__":
    main()
