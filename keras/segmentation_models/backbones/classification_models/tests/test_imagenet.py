import numpy as np
from skimage.io import imread
from keras.applications.imagenet_utils import decode_predictions

import sys
sys.path.insert(0, '..')

from classification_models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from classification_models import ResNeXt50, ResNeXt101
from classification_models import resnet
from classification_models import resnext


models_zoo = {
    'resnet18': {
        'model': ResNet18,
        'params': [
            {
                'input_shape': (224,224,3),
                'dataset': 'imagenet',
                'ground_truth': [(144, 0.5189058), (23, 0.17232688), (21, 0.098873824), (22, 0.03640686), (315, 0.023893135)],
                'preprocessing_function': lambda x:resnet.preprocess_input(x, (224, 224), True),
            }
        ]
    },
    
    'resnet34': {
        'model': ResNet34,
        'params': [
            {
                'input_shape': (224,224,3),
                'dataset': 'imagenet',
                'ground_truth': [(144, 0.88104683), (23, 0.031556014), (21, 0.024246644), (146, 0.022548646), (94, 0.0057696267)],
                'preprocessing_function': lambda x:resnet.preprocess_input(x, (224, 224), True),
            }
        ]
    },

    'resnet50': {
        'model': ResNet50,
        'params': [
            {
                'input_shape': (224, 224, 3),
                'dataset': 'imagenet',
                'ground_truth': [(21, 0.53156805), (144, 0.37913376), (23, 0.057184655), (146, 0.024926249), (22, 0.0015899206)],
                'preprocessing_function': lambda x: resnet.preprocess_input(x, (224, 224), True),
            },
        ]
    },

    'resnet101': {
        'model': ResNet101,
        'params': [
            {
                'input_shape': (224,224,3),
                'dataset': 'imagenet',
                'ground_truth': [(21, 0.96975815), (144, 0.016729029), (146, 0.00535842), (99, 0.0017561398), (22, 0.0010300555)],
                'preprocessing_function': lambda x:resnet.preprocess_input(x, (224, 224), True),
            }
        ]
    },

    'resnet152': {
        'model': ResNet152,
        'params': [
            {
                'input_shape': (224,224,3),
                'dataset': 'imagenet',
                'ground_truth': [(21, 0.59152377), (144, 0.2688002), (97, 0.0474935), (146, 0.035076432), (99, 0.014631907)],
                'preprocessing_function': lambda x:resnet.preprocess_input(x, (224, 224), True),
            }
        ]
    },

    'resnext50': {
        'model': ResNeXt50,
        'params': [
            {
                'input_shape': (224,224,3),
                'dataset': 'imagenet',
                'ground_truth': [(396, 0.97365075), (398, 0.0096320715), (409, 0.005558599), (438, 0.0028824762), (440, 0.0019731398)],
                'preprocessing_function': lambda x:resnext.preprocess_input(x, (224, 224)),
            }
        ]
    },

    'resnext101': {
        'model': ResNeXt101,
        'params': [
            {
                'input_shape': (224,224,3),
                'dataset': 'imagenet',
                'ground_truth': [(396, 0.95073587), (440, 0.016645206), (426, 0.004068849), (398, 0.0032844676), (392, 0.0022560472)],
                'preprocessing_function': lambda x:resnext.preprocess_input(x, (224, 224)),
            }
        ]
    },
}


def get_top(y, top=5):
    y = y.squeeze()
    idx = y.argsort()[::-1]
    top_idx = idx[:top]
    top_pred = y[top_idx]
    return list(zip(top_idx, top_pred))


def is_equal(gt, pr, eps=10e-5):

    for i in range(len(gt)):
        idx_gt, prob_gt = gt[i]
        idx_pr, prob_pr = pr[i]

        if idx_gt != idx_pr:
            return False

        if not np.allclose(prob_gt, prob_pr, atol=eps):
            return False
    return True


def test_model(model, preprocessing_func, sample, ground_truth):

    x = preprocessing_func(sample)
    x = np.expand_dims(x, 0)
    y = model.predict(x)

    print('[INFO]', decode_predictions(y))

    pred = get_top(y)
    if is_equal(pred, ground_truth):
        print('[INFO] Test passed...\n')
    else:
        print('[WARN] TEST FAILED...')
        print('[WARN] PREDICTION', pred)
        print('[WARN] GROUND TRUTH', ground_truth)
        print()


def main():

    path = ('../imgs/tests/seagull.jpg')
    img = imread(path)
    for model_type in models_zoo:
        for params in models_zoo[model_type]['params']:

            input_shape = params['input_shape']
            dataset = params['dataset']
            preprocessing_function = params['preprocessing_function']
            groud_truth = params['ground_truth']

            print('[INFO] Loading model {} with weights {}....'.format(model_type, dataset))
            model = models_zoo[model_type]['model']
            model = model(input_shape, weights=dataset, classes=1000)

            test_model(model, preprocessing_function, img, groud_truth)

if __name__ == '__main__':
    main()
