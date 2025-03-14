#!/bin/bash
echo "Processing events and sports data..."

# Create clean-data directory if it doesn't exist
mkdir -p clean-data

# Run the processing script
python src/data_processing/process_events_sports.py

echo "Processing complete! Check clean-data/events_sports.txt for the results." 