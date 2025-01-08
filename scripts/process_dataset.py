import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import csv
import json
from typing import List
from models.models import ReviewItem

def process_dataset(input_csv: str, output_json: str):
    reviews = []
    
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            review = ReviewItem(
                reviewId=row['review_id'],
                text=row['review'],
                polarity=row['sentiment']
            )
            reviews.append(review.model_dump())
    
    output_data = {
        "reviews": reviews
    }
    
    with open(output_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_data, jsonfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_dataset.py <input_csv> <output_json>")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    output_json = sys.argv[2]
    process_dataset(input_csv, output_json)
