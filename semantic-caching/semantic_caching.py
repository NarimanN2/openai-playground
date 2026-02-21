import json
import math
import os
import time
import uuid

import redis
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from scipy.spatial.distance import cosine

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url, decode_responses=True)

namespace = "chatgpt"
index_key = f"semantic_cache:index:{namespace}"

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5-nano")

prompts = [
    "Why the sky is blue?",
    "Explain to me why the color of the sky is blue?",
    "What is the population of Germany?",
]

def cosine_distance(a, b):
    distance = float(cosine(a, b))

    if math.isnan(distance):
        distance = 1.0

    return distance


def clear_namespace():
    keys = redis_client.smembers(index_key)

    if keys:
        redis_client.delete(*keys)

    redis_client.delete(index_key)


def find_best_match(query_embedding, score_threshold=0.05):
    # Important: this is a naive full-scan approach for tutorial purposes only.
    # It is not production-ready! use Redis vector search functionality in real systems.
    keys = redis_client.smembers(index_key)
    best_distance = None
    best_response = None

    for key in keys:
        entry = redis_client.hgetall(key)
        entry_embedding = json.loads(entry["embedding"])
        distance = cosine_distance(query_embedding, entry_embedding)

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_response = entry["response"]

    if best_distance is None or best_distance > score_threshold:
        return None

    return best_response


def save_entry(prompt, response, embedding):
    key = f"semantic_cache:entry:{namespace}:{uuid.uuid4()}"
    redis_client.hset(
        key,
        mapping={
            "prompt": prompt,
            "response": response,
            "embedding": json.dumps(embedding),
            "created_at": str(time.time())
        }
    )

    redis_client.sadd(index_key, key)


clear_namespace()

for prompt in prompts:
    query_embedding = embeddings.embed_query(prompt)
    cached_response = find_best_match(query_embedding)

    print(f"\nPrompt: {prompt}")

    if cached_response is not None:
        print(f"Cached Response: {cached_response}")

    else:
        response = llm.invoke(prompt)
        save_entry(prompt, response.content, query_embedding)

        print(f"API Response: {response.content}")
