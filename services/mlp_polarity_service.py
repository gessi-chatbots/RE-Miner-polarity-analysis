from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from models.models import ReviewItem
from services.polarity_service_base import PolarityService
import logging

class MLPPolarityService(PolarityService):
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
                self.logger.info("Loading pre-trained MLP model...")
                self.model = joblib.load('trained_mlp_model.joblib')
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
        
        # Create pipeline with TF-IDF and MLP
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        
        # MLP configuration
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('smote', SMOTE()),
            ('mlp', mlp_classifier)
        ])

        self.logger.info("Creating TF-IDF vectorizer and MLP pipeline...")
        # Train the model on all data
        self.logger.info("Training model on full dataset...")
        self.model.fit(texts, labels)
        
        self.logger.info("Saving trained model...")
        joblib.dump(self.model, 'trained_mlp_model.joblib')
        self.logger.info("Model training completed successfully") 