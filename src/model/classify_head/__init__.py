from src.model.classify_head.logistic_model import LogisticModel
from src.model.classify_head.moe_model import MoeModel, MoEWithCG
from src.model.classify_head.logistic import Logistic
from src.model.classify_head.sigmoid import Sigmoid
from src.model.classify_head.conditional_inference import ConditionalInference
from src.model.classify_head.mlp import MLP
from src.model.classify_head.hmcn import HMCN


def get_instance(name, paramters_dict):
    model = {
        'LogisticModel': LogisticModel,
        'MoeModel': MoeModel,
        'Logistic': Logistic,
        'Sigmoid': Sigmoid,
        'MoeWithCG': MoEWithCG,
        'ConditionalInference': ConditionalInference,
        'MLP': MLP,
        'HMCN': HMCN
    }[name]
    return model(**paramters_dict)
