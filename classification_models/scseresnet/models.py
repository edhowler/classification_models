from .builder import build_scseresnet
from ..resnet.builder import build_resnet
from ..utils import load_model_weights
from ..weights import weights_collection



def SCSEResNet18(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights:
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             repetitions=(2, 2, 2, 2),
                             classes=classes,
                             include_top=include_top,
                             block_type='basic')
        model.name = 'resnet18'
        load_model_weights(weights_collection, model, weights, classes, include_top)
        model.save_weights('resnet18_weights.h5')

    model_se = build_scseresnet(input_tensor=input_tensor,
                                input_shape=input_shape,
                                repetitions=(2, 2, 2, 2),
                                classes=classes,
                                include_top=include_top,
                                block_type='basic')

    if weights:
        model_se.load_weights('resnet18_weights.h5', by_name=True)
    return model_se


def SCSEResNet34(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights:
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             repetitions=(3, 4, 6, 3),
                             classes=classes,
                             include_top=include_top,
                             block_type='basic')
        model.name = 'resnet34'
        load_model_weights(weights_collection, model, weights, classes, include_top)
        model.save_weights('resnet34_weights.h5')

    model_se = build_scseresnet(input_tensor=input_tensor,
                                input_shape=input_shape,
                                repetitions=(3, 4, 6, 3),
                                classes=classes,
                                include_top=include_top,
                                block_type='basic')
    if weights:
        model_se.load_weights('resnet34_weights.h5', by_name=True)
    return model_se


def SCSEResNet50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights:
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             repetitions=(3, 4, 6, 3),
                             classes=classes,
                             include_top=include_top)
        model.name = 'resnet50'
        load_model_weights(weights_collection, model, weights, classes, include_top)
        model.save_weights('resnet50_weights.h5')

    if weights:
        model_se.load_weights('resnet50_weights.h5', by_name=True)
    return model_se


def SCSEResNet101(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights:
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             repetitions=(3, 4, 23, 3),
                             classes=classes,
                             include_top=include_top)
        model.name = 'resnet101'
        load_model_weights(weights_collection, model, weights, classes, include_top)
        model.save_weights('resnet101_weights.h5')

    if weights:
        model_se.load_weights('resnet101_weights.h5', by_name=True)
    return model_se


def SCSEResNet152(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights:
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             repetitions=(3, 8, 36, 3),
                             classes=classes,
                             include_top=include_top)
        model.name = 'resnet152'
        load_model_weights(weights_collection, model, weights, classes, include_top)
        model.save_weights('resnet152_weights.h5')

    if weights:
        model_se.load_weights('resnet152_weights.h5', by_name=True)
    return model_se
