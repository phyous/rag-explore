import logging
from typing import Any, Callable, Dict, List, Set, Tuple
import xml.etree.ElementTree as ET
import concurrent.futures
import time
from tqdm import tqdm

from src.helpers import create_client

def calculate_mrr(retrieved_links: List[str], correct_links: Set[str]) -> float:
    for i, link in enumerate(retrieved_links, 1):
        if link in correct_links:
            return 1 / i
    return 0

def evaluate_retrieval(retrieval_function: Callable, evaluation_data: List[Dict[str, Any]], db: Any) -> Tuple[float, float, float, float, List[float], List[float], List[float]]:
    precisions = []
    recalls = []
    mrrs = []
    
    for i, item in enumerate(tqdm(evaluation_data, desc="Evaluating Retrieval")):
        try:
            retrieved_chunks, _ = retrieval_function(item['question'], db)
            retrieved_links = [chunk['metadata'].get('chunk_link', chunk['metadata'].get('url', '')) for chunk in retrieved_chunks]
        except Exception as e:
            logging.error(f"Error in retrieval function: {e}")
            continue

        correct_links = set(item['correct_chunks'])
        
        true_positives = len(set(retrieved_links) & correct_links)
        precision = true_positives / len(retrieved_links) if retrieved_links else 0
        recall = true_positives / len(correct_links) if correct_links else 0
        mrr = calculate_mrr(retrieved_links, correct_links)
        
        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(evaluation_data)} items. Current Avg Precision: {sum(precisions) / len(precisions):.4f}, Avg Recall: {sum(recalls) / len(recalls):.4f}, Avg MRR: {sum(mrrs) / len(mrrs):.4f}")
    
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs

def process_single_evaluation(item: Dict[str, Any], answer_query_function: Callable, db: Any) -> Tuple[bool, str]:
    query = item['question']
    correct_answer = item['correct_answer']
    generated_answer = answer_query_function(query, db)
    
    prompt = f"""
    You are an AI assistant tasked with evaluating the correctness of answers to questions about Anthropic's documentation.
    
    Question: {query}
    
    Correct Answer: {correct_answer}
    
    Generated Answer: {generated_answer}
    
    Is the Generated Answer correct based on the Correct Answer? You should pay attention to the substance of the answer, and ignore minute details that may differ. 
    
    Small differences or changes in wording don't matter. If the generated answer and correct answer are saying essentially the same thing then that generated answer should be marked correct. 
    
    However, if there is any critical piece of information which is missing from the generated answer in comparison to the correct answer, then we should mark this as incorrect. 
    
    Finally, if there are any direct contradictions between the correect answer and generated answer, we should deem the generated answer to be incorrect.
    
    Respond in the following XML format:
    <evaluation>
    <content>
    <explanation>Your explanation here</explanation>
    <is_correct>true/false</is_correct>
    </content>
    </evaluation>
    """
    
    max_retries = 3
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            client = create_client()
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<evaluation>"}
                ],
                temperature=0,
                stop_sequences=["</evaluation>"]
            )
            
            response_text = response.content[0].text
            evaluation = ET.fromstring(response_text)
            is_correct = evaluation.find('is_correct').text.lower() == 'true'
            return is_correct, query
            
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Rate limit hit, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            logging.error(f"Error processing query '{query}': {str(e)}")
            return False, query

def evaluate_end_to_end(answer_query_function: Callable, db: Any, eval_data: List[Dict[str, Any]], parallelism: int = 5) -> Tuple[float, List[bool]]:
    """
    Evaluate the end-to-end performance of a question answering system.
    
    Args:
        answer_query_function: Function that takes a query and db and returns an answer
        db: The vector database instance
        eval_data: List of evaluation data items
        parallelism: Number of parallel evaluations to run (default: 5)
    
    Returns:
        Tuple of (accuracy, list of results)
    """
    results = []
    correct_answers = 0
    total_questions = len(eval_data)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_item = {
            executor.submit(process_single_evaluation, item, answer_query_function, db): i 
            for i, item in enumerate(eval_data)
        }
        
        with tqdm(total=total_questions, desc="Evaluating End-to-End") as pbar:
            for future in concurrent.futures.as_completed(future_to_item):
                is_correct, query = future.result()
                if is_correct:
                    correct_answers += 1
                results.append(is_correct)
                
                item_index = future_to_item[future]
                current_accuracy = correct_answers / (item_index + 1)
                
                pbar.update(1)
                if (item_index + 1) % 10 == 0:
                    logging.info(f"Processed {item_index + 1}/{total_questions} questions. Current Accuracy: {current_accuracy:.4f}")
    
    accuracy = correct_answers / total_questions
    return accuracy, results