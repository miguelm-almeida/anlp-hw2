# Team Contributions

## Data Annotation
- Prith Sharma: Annotator 1 for IAA and also annotated Events and Sports
- Sudeshna Merugu: Annotator 2 for IAA also annotated General and History
- Miguel Almeida: Implemented IAA agreement calculation and also annotated Music and Culture

## Implementation
- Sudeshna Merugu:
  - Created the vector database by chunking the data and generating embeddings (Corpus processing) and worked on RAG QA
  - Implemented batch processing script for CSV input inference
  - Developed `main.py` for batch and single input inference

- Prith Sharma:
  - Implemented RAG Retriever
  - Conducted closed book evaluation (RAG + LLMs vs LLMs only)
  - Computed metrics: Exact match, F1, semantic similarity, answer recall
  
- Miguel Almeida:
  - Developed Lightweight QA for concise answer generation using retrieved context
  - Implemented IAA calculation
  - Integrated report writing and final deliverables (code, data, README files)
