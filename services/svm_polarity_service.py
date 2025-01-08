from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from models.models import ReviewItem
from services.polarity_service_base import PolarityService
import logging

class SVMPolarityService(PolarityService):
    def __init__(self):
        self.model = None
        self.vectorizer = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_polarity(self, reviews: List[ReviewItem], threshold: float = None) -> List[ReviewItem]:
        self.logger.info(f"Starting polarity analysis for {len(reviews)} reviews")
        
        # Check if model exists
        if self.model is None:
            try:
                self.logger.info("Loading pre-trained SVM model...")
                self.model = joblib.load('trained_svm_model.joblib')
            except FileNotFoundError:
                self.logger.error("No trained model found!")
                raise ValueError("No trained model found. Please train the model first.")
        
        # Extract text from reviews
        texts = [review.text for review in reviews]
        
        self.logger.info("Making predictions...")
        predictions = self.model.predict(texts)
        
        self.logger.info("Updating review sentiments with predictions")
        # Update sentiment values in the ReviewItems
        for review, prediction in zip(reviews, predictions):
            review.polarity = prediction
            
        self.logger.info("Polarity analysis completed")
        return reviews
    
    def train_model(self, data: List[ReviewItem]):
        self.logger.info(f"Starting model training with {len(data)} samples")
        
        # Extract text and labels from ReviewItems
        texts = [review.text for review in data]
        labels = [review.polarity for review in data]
        
        # Create pipeline with TF-IDF and SVM
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('smote', SMOTE()),
            ('svm', SVC(kernel='linear'))
        ])

        self.logger.info("Creating TF-IDF vectorizer and SVM pipeline...")
        # Train the model on all data
        self.logger.info("Training model on full dataset...")
        self.model.fit(texts, labels)
        
        self.logger.info("Saving trained model...")
        joblib.dump(self.model, 'trained_svm_model.joblib')
        self.logger.info("Model training completed successfully") 