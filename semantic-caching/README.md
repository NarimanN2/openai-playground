# Semantic Caching

A simple semantic cache example that uses Redis with LangChain embeddings and ChatGPT API. The cache stores responses based on semantic similarity to avoid paying for the same prompts again and again while also making responses faster.

## Pre-requisites

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

Generate an OpenAI API key and set the OPENAI_API_KEY environment variable:

```bash
export OPENAI_API_KEY=...
```

Make sure docker is installed and then run Redis locally using:

```bash
docker run -p 6379:6379 redis
```

## Run

```bash
python semantic_caching.py
```