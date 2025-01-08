from abc import ABC, abstractmethod
from typing import List
from models.models import ReviewItem

class PolarityService(ABC):
    @abstractmethod
    def analyze_polarity(self, reviews: List[ReviewItem], threshold: float = 0.2) -> List[ReviewItem]:
        pass 
    
    @abstractmethod
    def train_model(self, data: List[ReviewItem]):
        pass 