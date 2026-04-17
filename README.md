# Domain-aware RAG system for industrial maintenance and diagnostics

for demo purposes, the knowledge base is limited to the pdf service manual retrievable from the followiing link:
https://library.e.abb.com/public/a44d07ce27e7665e85257ccb00539304/3ADW000195_F.pdf

Knowledge base description: [./data/raw]
This document is a technical guide focused on the principles, operation, and practical applications of variable speed drives (VSDs) in industrial systems. It provides a structured overview of how electric drives are used to control motor speed, improve energy efficiency, and optimize process performance.
The guide covers key topics such as:
- fundamentals of electric motors and drive systems
- speed and torque control strategies
- typical industrial use cases (e.g., pumps, fans, conveyors)
- energy efficiency considerations and cost savings
- common operational issues and best practices
Although it is partially educational in nature, the document includes real-world engineering concepts and practical insights that are commonly encountered in industrial environments.

Why this document is used in the project
This PDF is used as part of the knowledge base to simulate a realistic industrial documentation scenario.
It provides:
- domain-specific terminology
- structured technical explanations
- context for troubleshooting and system understanding
This makes it suitable for testing Retrieval-Augmented Generation (RAG) pipelines in a technical setting, even with a limited number of documents.

### Qdrant Vector DB

pull qdrant image and container run + mounting:
```bash
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

if you restart the pc, you can just re run the container.

Qdrant UI available at: http://localhost:6333/dashboard