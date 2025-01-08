from enums.service_enum import PolarityServiceType
from services.default_polarity_service import DefaultPolarityService
from services.svm_polarity_service import SVMPolarityService
from services.mlp_polarity_service import MLPPolarityService
from services.polarity_service_base import PolarityService

class PolarityServiceFactory:
    @staticmethod
    def get_service(service_type: str) -> PolarityService:
        try:
            service_enum = PolarityServiceType(service_type)
        except ValueError:
            available_services = ", ".join(sorted(enum.value for enum in ServiceEnum))
            raise ValueError(
                f"Invalid polarity service type: '{service_type}'. "
                f"Available services are: {available_services}"
            )
        
        if service_enum == PolarityServiceType.DEFAULT:
            return DefaultPolarityService()
        if service_enum == PolarityServiceType.SVM:
            return SVMPolarityService() 
        if service_enum == PolarityServiceType.MLP:
            return MLPPolarityService() 
        logging.error(f"Invalid service type: {service_type}")
        raise ValueError(f"Invalid service type: {service_type}")
