from .builder import build_resnet
from ..utils import load_model_weights
from ..weights import weights_collection


def ResNet18(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(2, 2, 2, 2),
                         classes=classes,
                         include_top=include_top,
                         block_type='basic')
    model.name = 'resnet18'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet34(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top,
                         block_type='basic')
    model.name = 'resnet34'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top)
    model.name = 'resnet50'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet101(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 23, 3),
                         classes=classes,
                         include_top=include_top)
    model.name = 'resnet101'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet152(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 8, 36, 3),
                         classes=classes,
                         include_top=include_top)
    model.name = 'resnet152'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model
