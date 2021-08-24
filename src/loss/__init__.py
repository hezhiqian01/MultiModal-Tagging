from src.loss.loss import CrossEntropyLoss
from src.loss.loss import SoftmaxLoss, DBLoss, HMCNLoss


def get_instance(name, paramters_dict):
    model = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'SoftmaxLoss': SoftmaxLoss,
        'DBLoss': DBLoss,
        'HMCNLoss': HMCNLoss
    }[name]
    return model(**paramters_dict)

