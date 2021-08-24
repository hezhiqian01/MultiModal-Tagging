from src.model.fusion_head.fusion_se import SE
from src.model.fusion_head.fusion_mlgcn import MLGCN
from src.model.fusion_head.fusion_cg import ContextGating
from src.model.fusion_head.fusion_transformer import FusionTrm


def get_instance(name, paramters):
    model = {
        'SE': SE,
        'MLGCN': MLGCN,
        'CG': ContextGating,
        "TRM": FusionTrm
    }[name]
    return model(**paramters)
