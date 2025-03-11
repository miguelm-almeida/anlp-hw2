#!/bin/bash
echo "Generating answers for test questions..."

# Create output directory
mkdir -p system_outputs

# Run RAG system
python src/rag_system/rag_system.py --questions data/test/questions.txt --output system_outputs/system_output_1.json

echo "Answer generation complete!" 