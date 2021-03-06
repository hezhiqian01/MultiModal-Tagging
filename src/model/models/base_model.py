class BaseModel():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, inputs, is_training):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError
