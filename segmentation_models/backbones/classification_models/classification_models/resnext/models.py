from .builder import build_resnext
from ..utils import load_model_weights
from ..weights import weights_collection


def ResNeXt50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnext(input_tensor=input_tensor,
                         input_shape=input_shape,
                         first_block_filters=128,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top)
    model.name = 'resnext50'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNeXt101(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnext(input_tensor=input_tensor,
                         input_shape=input_shape,
                         first_block_filters=128,
                         repetitions=(3, 4, 23, 3),
                         classes=classes,
                         include_top=include_top)
    model.name = 'resnext101'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model

