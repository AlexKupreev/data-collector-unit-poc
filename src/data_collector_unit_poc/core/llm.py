"""LLM-related functionality"""
import json
from datetime import datetime

from openai import OpenAI

class NoRelevantDataFoundError(Exception):
    ...

def openai_llm(client: OpenAI, prompt: str, model_name: str = 'gpt-4o-mini') -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def rag(
    query: str,
    prompt_fn: callable,
    llm_fn: callable,
    search_fn: callable,
    num_results: int = 8,
    query_vector: list | None = None,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None
) -> str:
    """Return answer built using RAG"""

    if query_vector:
        search_results = search_fn(
            query=query_vector,
            num_results=num_results,
            start_dt=start_dt,
            end_dt=end_dt
        )
    else:
        search_results = search_fn(
            query=query,
            num_results=num_results,
            start_dt=start_dt,
            end_dt=end_dt
        )

    if not search_results:
        raise NoRelevantDataFoundError("No relevant results found.")
    prompt = prompt_fn(query, search_results)
    answer = llm_fn(prompt)
    return answer

def extract_from_llm_output(raw_rag_result: str) -> list[dict]:
    """Extract JSON from the LLM output"""
    json_str = raw_rag_result.strip('`').removeprefix("json").strip()
    return json.loads(json_str)