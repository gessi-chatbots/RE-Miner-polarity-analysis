from fastapi import APIRouter, Query
from models.models import PolarityRequest, TrainModelRequest
from services.polarity_service_factory import PolarityServiceFactory
from services.polarity_service import PolarityService
from enums.service_enum import PolarityServiceType
import logging

router = APIRouter()

@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.post("/analyze-polarity")
async def analyze_polarity(
    request: PolarityRequest,
    polarity_service: PolarityServiceType = Query(..., alias="polarity-service")
):
    service = PolarityServiceFactory.get_service(polarity_service.value)
    analyzed_reviews = service.analyze_polarity(request.reviews)
    return {"reviews": analyzed_reviews} 

@router.post("/train-model")
async def train_model(
    request: TrainModelRequest,
    polarity_service: PolarityServiceType = Query(..., alias="polarity-service")
):
    service = PolarityServiceFactory.get_service(polarity_service.value)
    logging.info(f"Service type: {type(service)}")
    filtered_reviews = [review for review in request.reviews if review.polarity != 'N/A']
    service.train_model(filtered_reviews)
    return {"status": "ok"}
