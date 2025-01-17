
Following along in:
https://github.com/anthropics/anthropic-cookbook/blob/main/skills/retrieval_augmented_generation/guide.ipynb

# 1. Basic RAG
A basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

1. Chunk documents by heading - containing only the content from each subheading
2. Embed each document
3. Use Cosine similarity to retrieve documents in order to answer query

Run the following command to execute the basic RAG pipeline:
> python rag_1_basic.py
```
Evaluating Retrieval:  99%|█████████████████████████▋| 99/100 [00:13<00:00,  8.19it/s]Processed 100/100 items. Current Avg Precision: 0.4033, Avg Recall: 0.6450, Avg MRR: 0.7233
Evaluating End-to-End: 100%|████████████████████████| 100/100 [02:40<00:00,  1.60s/it]
Detailed results saved to evaluation/csvs/evaluation_results_one.csv
Average Precision: 0.4033
Average Recall: 0.6450
Average MRR: 0.7233
Average F1: 0.4963
End-to-End Accuracy: 0.7200
Evaluation complete. Results saved to evaluation_results_one.json, evaluation_results_one.csv
```

# 2. RAG with Summarized Retrieval
Instead of embedding chunks directly from the documents, we'll create a concise summary for each chunk and use this summary along with the original content in our embedding process.

> python rag_2_summarized_retrieval.py
```
Processed 100/100 items. Current Avg Precision: 0.3983, Avg Recall: 0.6383, Avg MRR: 0.7417
Evaluating Retrieval: 100%|█████████████████████████████████████████████████████████████████████████| 100/100 [00:21<00:00,  4.66it/s]
Evaluating End-to-End: 100%|████████████████████████████████████████████████████████████████████████| 100/100 [02:05<00:00,  1.26s/it]
Detailed results saved to evaluation_results_detailed.csv
Average Precision: 0.3983
Average Recall: 0.6383
Average MRR: 0.7417
Average F1: 0.4906
End-to-End Accuracy: 0.7600
```

# 3. RAG with LLM based re-ranking

In this final enhancement to our retrieval system, we introduce a reranking step to further improve the relevance of the retrieved documents. This approach leverages Claude's power to better understand the context and nuances of both the query and the retrieved documents.

> python rag_3_llm_reranking.py

```

```