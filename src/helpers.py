import os
import json
import anthropic
from src.db.summary_index_vetctor_db import SummaryIndexedVectorDB
from src.db.vector_db import VectorDB

def create_client():
    # Load credentials from json file
    try:
        with open('credentials.json', 'r') as f:
            credentials = json.load(f)
            api_key = credentials.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("anthropic_api_key not found in credentials.json")
    except FileNotFoundError:
        raise FileNotFoundError("credentials.json file not found. Please create one with your Anthropic API key")

    client = anthropic.Anthropic(
        api_key=api_key,
    )

    return client

def init_vector_db(collection_name="anthropic_docs"):
    """Initialize and return a VectorDB instance."""
    try:
        with open('credentials.json', 'r') as f:
            credentials = json.load(f)
            api_key = credentials.get('VOYAGE_API_KEY')
            if not api_key:
                raise ValueError("voyage_api_key not found in credentials.json")
    except FileNotFoundError:
        raise FileNotFoundError("credentials.json file not found. Please create one with your Voyage API key")

    return VectorDB(collection_name, api_key=api_key)

def init_summary_index_vector_db(collection_name="anthropic_docs_v2"):
    """Initialize and return a VectorDB instance."""
    try:
        with open('credentials.json', 'r') as f:
            credentials = json.load(f)
            api_key = credentials.get('VOYAGE_API_KEY')
            if not api_key:
                raise ValueError("voyage_api_key not found in credentials.json")
    except FileNotFoundError:
        raise FileNotFoundError("credentials.json file not found. Please create one with your Voyage API key")

    return SummaryIndexedVectorDB(collection_name, api_key=api_key)