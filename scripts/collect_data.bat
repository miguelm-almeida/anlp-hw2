@echo off
echo Starting data collection process...

:: Create raw data directory
mkdir raw_data 2>nul

:: Run web scraper
echo Running web scraper...
python src/data_collection/scraper.py

:: Run PDF processor
echo Running PDF processor...
python src/data_collection/pdf_processor.py

:: Run document preprocessor
echo Running document preprocessor...
python src/data_collection/document_preprocessor.py

echo Data collection complete! 