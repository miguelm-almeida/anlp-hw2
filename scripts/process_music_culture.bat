@echo off
echo Processing music and culture data...

:: Create clean-data directory if it doesn't exist
mkdir clean-data 2>nul

:: Run the processing script
python src/data_processing/process_music_culture.py

echo Processing complete! Check clean-data/music_culture.txt for the results. 