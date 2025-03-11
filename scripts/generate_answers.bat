@echo off
echo Generating answers for test questions...

:: Create output directory
mkdir system_outputs 2>nul

:: Run RAG system
python src/rag_system/rag_system.py --questions data/test/questions.txt --output system_outputs/system_output_1.json

echo Answer generation complete! 