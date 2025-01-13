from pydantic import BaseModel
from typing import List

class ReviewItem(BaseModel):
    reviewId: str
    text: str
    polarity: str | None = None

class PolarityRequest(BaseModel):
    reviews: List[ReviewItem] 

class TrainModelRequest(BaseModel):
    reviews: List[ReviewItem]

class SingleReviewRequest(BaseModel):
    text: str