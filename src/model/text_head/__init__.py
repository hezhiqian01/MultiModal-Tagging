from src.model.text_head.bert_model import BERT
from src.model.text_head.albert_model import ALBERT


def get_instance(name, paramters):
    model = {
        'BERT': BERT,
        'ALBERT': ALBERT,
    }[name]
    return model(**paramters)
