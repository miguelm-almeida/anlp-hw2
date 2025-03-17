# Retrieval Augmented Generation (RAG) System for Pittsburgh and CMU Q&A

This repository contains our implementation for the CMU Advanced NLP Assignment 2, where we built a Retrieval Augmented Generation system to answer questions about Pittsburgh and CMU.

## Team Members
- Prith Sharma: `priths`
- Sudeshna Merugu: `sudeshnm`
- Miguel Almeida: `malmeida`

## Project Overview

This project implements a Retrieval Augmented Generation (RAG) system to answer questions about Pittsburgh and Carnegie Mellon University. The system combines retrieval-based methods with language models to provide accurate and contextually relevant answers.

We evaluate three different language models in both closed book (without RAG) and open book (with RAG) configurations:
- Flan-T5 (google/flan-t5-small)
- DistilBERT (distilbert/distilbert-base-uncased-distilled-squad)
- Phi-2 (microsoft/phi-2)

## Repository Structure

```bash
├── RAG/                           # Main RAG implementation
│   ├── main.py                    # Main execution script
│   ├── lightweight_qa.py          # QA model implementations
│   ├── rag_retriever.py          # RAG retrieval component
│   ├── rag_question_answerer.py  # RAG question answering component
│   ├── batch_inference.py        # Batch processing script
│   ├── corpus_processor.py       # Document corpus processing
│   ├── closed_book_evaluation.ipynb # Closed book evaluation script
│   ├── metrics_evaluation.py     # Metrics calculation script
│   ├── empty.py                  # Empty file placeholder
│   ├── RAG.ipynb                # RAG implementation notebook
│   ├── requirements.txt         # RAG-specific dependencies
│   ├── corpus.txt              # Corpus for RAG retrieval
│   └── data/                   # Processed data for RAG
│         └── chunks.json        
│         └── embeddings.json
│
├── test/                       # Test data
│   ├── test_questions.txt     # Test questions
│   └── test_answers.txt       # Ground truth answers
│
├── Metrics and IAA/           # Metrics and evaluation
│   ├── Metrics_IAA.ipynb     # Metrics calculation notebook
│   ├── phi2_1500.csv         # Phi-2 model results
│   ├── distilbert_1500.csv   # DistilBERT model results
│   ├── flant5_1500.csv       # Flan-T5 model results
│   ├── merged_qa_distilbert.csv  # Merged results for DistilBERT
│   ├── merged_qa_flant5.csv     # Merged results for Flan-T5
│   ├── merged_qa_phi2.csv       # Merged results for Phi-2
│   ├── empty.py                 # Empty file placeholder
│   └── annotated_qsns_final.csv # Final annotated questions
│
├── submission_test_files/
│              └── system_outputs/
│                        └── system_output_1.json # Test submission files
│                        └── system_output_2.json # Test submission files
│                        └── system_output_3.json # Test submission files
│              ├── new_test_distilbert.csv
│              ├── new_test_flant5.csv
│              ├── new_test_phi2.csv
│              └── qa_links.txt
├── train/                    # Training data
│   ├── train_questions.txt     # Test questions
│   └── train_answers.txt       # Ground truth answers
├── src/                     # Source code
│   ├── data_processing/    # Data processing scripts
│   ├── data_collection/   # Data collection scripts
│   └── rag_system/       # RAG system implementation
│
├── raw-data/              # Raw data files
├── clean-data/           # Cleaned data files
├── annotated-data/      # Annotated data files
├── scripts/            # Utility scripts
├── requirements.txt   # Python dependencies
├── WRITEUP.md        # Project write-up and analysis
├── README.md         # Project documentation
├── contributions.md  # Team member contributions
└── github_url.txt   # Repository URL
```


## Installation and Setup

1. Clone this repository:
```bash
git clone https://github.com/miguelm-almeida/anlp-hw2.git
cd anlp-hw2
```

2. Download NLTK resources (for metrics calculation):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

3. Follow the steps mentioned in the RAG.ipynb file for the RAG implementation. 

4. Follow the steps mentioned in the Metrics_IAA.ipynb file to get the metrics and IAA values.

5. Run the RAG/closed_book_evaluation.ipynb to get the metrics for closed book evaluation. 


## Evaluation Metrics

We use four metrics to evaluate our QA system:

- **EM (Exact Match)**: Measures exact matches between predicted and ground truth answers
- **F1 Score**: Word-level F1 score between predicted and ground truth answers
- **SS (Semantic Similarity)**: Cosine similarity between sentence embeddings
- **AR (Answer Recall)**: Proportion of ground truth words in predicted answers


## System Architecture

Our RAG system consists of three main components:

1. **Document Retriever**: Uses semantic search to find relevant text chunks
2. **Context Builder**: Constructs context from retrieved chunks
3. **Question Answerer**: Generates answers using language models with context

The system supports three language models:
- **Flan-T5**: Sequence-to-sequence model for instruction-following tasks
- **DistilBERT**: Extractive QA model for identifying answer spans
- **Phi-2**: Compact but powerful causal language model

## Troubleshooting

- **Model Downloads**: Use `huggingface-cli login` for authentication errors
- **GPU Memory**: Adjust batch size if encountering memory issues
- **Path Issues**: Run commands from correct directory as shown in usage examples

## Acknowledgments

This project was developed as part of the Advanced NLP (11-711) course at Carnegie Mellon University.