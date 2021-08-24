from src.model.video_head.nextvlad import NeXtVLAD
from src.model.video_head.hlstm import HLSTM
from src.model.video_head.vtn import VTN


def get_instance(name, paramters_dict):
    model = {
        'NeXtVLAD': NeXtVLAD,
        "HLSTM": HLSTM,
        "VTN": VTN
    }[name]
    return model(**paramters_dict)
