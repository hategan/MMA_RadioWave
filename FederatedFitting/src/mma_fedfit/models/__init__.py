from typing import Dict
from .model import Model
from .afterglopy import Afterglowpy
from .fake import Fake


_MODELS: Dict[str, type] = {}


def get_model(name: str) -> Model:
    return _MODELS[name]()


def register_model(name: str, model_cls: type) -> None:
    _MODELS[name] = model_cls


register_model('afterglowpy', Afterglowpy)
register_model('fake', Fake)