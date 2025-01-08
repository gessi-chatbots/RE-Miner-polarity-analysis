from textblob import TextBlob
from typing import List
from models.models import ReviewItem
from services.polarity_service_base import PolarityService
import logging

class DefaultPolarityService(PolarityService):
    def analyze_polarity(self, reviews: List[ReviewItem], threshold: float = 0.2) -> List[ReviewItem]:
        for review in reviews:
            analysis = TextBlob(review.text)
            polarity_score = analysis.sentiment.polarity
            if polarity_score > threshold:
                review.polarity = "positive"
            elif polarity_score < -threshold:
                review.polarity = "negative"
            else:
                review.polarity = "neutral"
        return reviews 
    
    def train_model(self, data: List[ReviewItem]):
        logging.info("TextBlob sentiment analyzer does not require training")
        return