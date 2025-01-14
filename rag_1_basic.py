import json
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
from typing import Callable, List, Dict, Any, Tuple, Set

from src.eval.metrics import evaluate_end_to_end, evaluate_retrieval
from src.helpers import create_client, init_vector_db

client = create_client()

# Load the evaluation dataset
with open('evaluation/docs_evaluation_dataset.json', 'r') as f:
    eval_data = json.load(f)

# Load the Anthropic documentation
with open('data/anthropic_docs.json', 'r') as f:
    anthropic_docs = json.load(f)

# Initialize the VectorDB
db = init_vector_db()
db.load_data(anthropic_docs)

def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db):
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
    response = client.messages.create(
        model="claude-3.5-haiku-latest",
        max_tokens=2500,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.content[0].text

def preview_json(file_path, num_items=3):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        if isinstance(data, list):
            preview_data = data[:num_items]
        elif isinstance(data, dict):
            preview_data = dict(list(data.items())[:num_items])
        else:
            print(f"Unexpected data type: {type(data)}. Cannot preview.")
            return
        
        print(f"Preview of the first {num_items} items from {file_path}:")
        print(json.dumps(preview_data, indent=2))
        print(f"\nTotal number of items: {len(data)}")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

preview_json('evaluation/docs_evaluation_dataset.json')

# import pandas as pd

# avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs = evaluate_retrieval(retrieve_base, eval_data, db)
# e2e_accuracy, e2e_results = evaluate_end_to_end(answer_query_base, db, eval_data)

# # Create a DataFrame
# df = pd.DataFrame({
#     'question': [item['question'] for item in eval_data],
#     'retrieval_precision': precisions,
#     'retrieval_recall': recalls,
#     'retrieval_mrr': mrrs,
#     'e2e_correct': e2e_results
# })

# # Save to CSV
# df.to_csv('evaluation/csvs/evaluation_results_detailed.csv', index=False)
# print("Detailed results saved to evaluation/csvs/evaluation_results_one.csv")

# # Print the results
# print(f"Average Precision: {avg_precision:.4f}")
# print(f"Average Recall: {avg_recall:.4f}")
# print(f"Average MRR: {avg_mrr:.4f}")
# print(f"Average F1: {f1:.4f}")
# print(f"End-to-End Accuracy: {e2e_accuracy:.4f}")

# # Save the results to a file
# with open('evaluation/json_results/evaluation_results_one.json', 'w') as f:
#     json.dump({
#         "name": "Basic RAG",
#         "average_precision": avg_precision,
#         "average_recall": avg_recall,
#         "average_f1": f1,
#         "average_mrr": avg_mrr,
#         "end_to_end_accuracy": e2e_accuracy
#     }, f, indent=2)

# print("Evaluation complete. Results saved to evaluation_results_one.json, evaluation_results_one.csv")