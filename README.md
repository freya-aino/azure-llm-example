
# Agentic System for Tool Assisted Chat

## Setup

1) Install `uv`.

2) Fill out a `.env` file or set the environment Variables with the following keys (Standard Azure Environment Variables)

```
AZURE_AI_SEARCH_API_KEY=
AZURE_AI_SEARCH_INDEX_NAME=
AZURE_AI_SEARCH_SERVICE_NAME=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
```

3) Make sure that the following indecies are `Azure AI Search` keys are set (or change them in the Code):

```
chunk: The text excerpt from the document
title: The name of the Document (source)
```

## Install

> Initialize venv

```cmd
uv venv
```

> activate virtual env under ./.venv

> Install

```cmd
uv sync
```

## Run

```cmd
python test.py
```