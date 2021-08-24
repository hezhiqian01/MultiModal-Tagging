from src.model.models.nextvlad_bert import NextVladBERT
from src.model.models.fusion_model import FusionModel


def get_instance(name, paramters):
    model = {
        "NextVladBERT": NextVladBERT,
        "Fusion": FusionModel
    }[name]
    return model(paramters)
