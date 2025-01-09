
# 1. Basic RAG
A basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

1. Chunk documents by heading - containing only the content from each subheading
2. Embed each document
3. Use Cosine similarity to retrieve documents in order to answer query

Run the following command to execute the basic RAG pipeline:
> python rag_1_basic.py

# 2. RAG with Vector Database