import json
import concurrent.futures
import time
from anthropic import Anthropic
from tqdm import tqdm
from src.helpers import create_client

def process_single_summary(doc: dict, knowledge_base_context: str) -> dict:
    prompt = f"""
    You are tasked with creating a short summary of the following content from Anthropic's documentation. 

    Context about the knowledge base:
    {knowledge_base_context}

    Content to summarize:
    Heading: {doc['chunk_heading']}
    {doc['text']}

    Please provide a brief summary of the above content in 2-3 sentences. The summary should capture the key points and be concise. We will be using it as a key part of our search pipeline when answering user queries about this content. 

    Avoid using any preamble whatsoever in your response. Statements such as 'here is the summary' or 'the summary is as follows' are prohibited. You should get straight into the summary itself and be concise. Every word matters.
    """

    max_retries = 3
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            client = create_client()
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=150,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            summary = response.content[0].text.strip()
            return {
                "chunk_link": doc["chunk_link"],
                "chunk_heading": doc["chunk_heading"],
                "text": doc["text"],
                "summary": summary
            }
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limit hit, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            print(f"Error processing chunk '{doc['chunk_heading']}': {str(e)}")
            # Return original doc with empty summary on failure
            return {**doc, "summary": ""}

def generate_summaries(input_file, output_file, parallelization_factor=8):
    """
    Generate summaries for documents with parallel processing and rate limit handling.
    
    Args:
        input_file: Path to input JSON file containing documents
        output_file: Path to output JSON file for summarized documents
        parallelization_factor: Number of parallel summary generations to run (default: 8)
    """
    # Load the original documents
    with open(input_file, 'r') as f:
        docs = json.load(f)

    # Prepare the context about the overall knowledge base
    knowledge_base_context = "This is documentation for Anthropic's, a frontier AI lab building Claude, an LLM that excels at a variety of general purpose tasks. These docs contain model details and documentation on Anthropic's APIs."

    summarized_docs = []
    total_docs = len(docs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelization_factor) as executor:
        future_to_doc = {
            executor.submit(process_single_summary, doc, knowledge_base_context): i 
            for i, doc in enumerate(docs)
        }
        
        with tqdm(total=total_docs, desc="Generating summaries") as pbar:
            for future in concurrent.futures.as_completed(future_to_doc):
                try:
                    summarized_doc = future.result()
                    summarized_docs.append(summarized_doc)
                    pbar.update(1)
                except Exception as e:
                    print(f"Unexpected error in summary generation: {str(e)}")
                    continue

    # Save the summarized documents to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(summarized_docs, f, indent=2)

    print(f"Summaries generated and saved to {output_file}")

generate_summaries('data/anthropic_docs.json', 'data/anthropic_summary_indexed_docs.json')