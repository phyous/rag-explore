import json
import concurrent.futures
import time

import pandas as pd

from anthropic import Anthropic
from tqdm import tqdm
from src.eval.metrics import evaluate_end_to_end, evaluate_retrieval
from src.db.helpers import retrieve_base
from src.helpers import create_client, init_summary_index_vector_db

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

# 1. Genrate summaries using Anthropic
#generate_summaries('data/anthropic_docs.json', 'data/anthropic_summary_indexed_docs.json')

# 2.Enhanced Retrieval Using Summary-Indexed Embeddings
'''
In this section, we implement the retrieval process using our new summary-indexed vector database. This approach leverages the enhanced embeddings we created, which incorporate document summaries along with the original content.

Key aspects of this updated retrieval process:

We search the vector database using the query embedding, retrieving the top k most similar documents.
For each retrieved document, we include the chunk heading, summary, and full text in the context provided to the LLM.
This enriched context is then used to generate an answer to the user's query.
By including summaries in both the embedding and retrieval phases, we aim to provide the LLM with a more comprehensive and focused context. This could potentially lead to more accurate and relevant answers, as the LLM has access to both a concise overview (the summary) and the detailed information (the full text) for each relevant document chunk.
'''
def retrieve_level_two(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n <document> \n {chunk['chunk_heading']}\n\nText\n {chunk['text']} \n\nSummary: \n {chunk['summary']} \n </document> \n" #show model all 3 items
    return results, context

def answer_query_level_two(query, db):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query: 
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. 
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    client = create_client()
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=2500,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.content[0].text

# Initialize the SummaryIndexedVectorDB
level_two_db = init_summary_index_vector_db()
level_two_db.load_data('data/anthropic_summary_indexed_docs.json')

# Run the evaluations
# Load the evaluation dataset
with open('evaluation/docs_evaluation_dataset.json', 'r') as f:
    eval_data = json.load(f)

# Load the Anthropic documentation
with open('data/anthropic_docs.json', 'r') as f:
    anthropic_docs = json.load(f)

avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs  = evaluate_retrieval(retrieve_level_two, eval_data, level_two_db)
e2e_accuracy, e2e_results = evaluate_end_to_end(answer_query_level_two, level_two_db, eval_data)

# Create a DataFrame
df = pd.DataFrame({
    'question': [item['question'] for item in eval_data],
    'retrieval_precision': precisions,
    'retrieval_recall': recalls,
    'retrieval_mrr': mrrs,
    'e2e_correct': e2e_results
})

# Save to CSV
df.to_csv('evaluation/csvs/evaluation_results_detailed_level_two.csv', index=False)
print("Detailed results saved to evaluation_results_detailed.csv")

# Print the results
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average MRR: {avg_mrr:.4f}")
print(f"Average F1: {f1:.4f}")
print(f"End-to-End Accuracy: {e2e_accuracy:.4f}")

# Save the results to a file
with open('evaluation/json_results/evaluation_results_level_two.json', 'w') as f:
    json.dump({
        "name": "Summary Indexing",
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": f1,
        "average_mrr": avg_mrr,
        "end_to_end_accuracy": e2e_accuracy
    }, f, indent=2)

print("Evaluation complete. Results saved to evaluation_results_level_two.json, evaluation_results_detailed_level_two.csv")