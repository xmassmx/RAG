"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = "rag"
JOB_TYPE = "fetch"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="massm/rag/vector_store:latest",
    chat_prompt_artifact="massm/rag/chat_prompt:latest",
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name="gpt-3.5-turbo",
    eval_model="gpt-3.5-turbo",
    eval_artifact="massm/rag/generated_examples:v0",
)