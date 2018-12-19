weights_collection = [

    # ResNet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5',
        'name': 'resnet18_imagenet_1000.h5',
        'md5': '64da73012bb70e16c901316c201d9803',
    },

    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5',
        'name': 'resnet18_imagenet_1000.h5',
        'md5': '318e3ac0cd98d51e917526c9f62f0b50',
    },

    # ResNet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
        'name': 'resnet34_imagenet_1000.h5',
        'md5': '2ac8277412f65e5d047f255bcbd10383',
    },

    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582',
    },

    # ResNet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000.h5',
        'name': 'resnet50_imagenet_1000.h5',
        'md5': 'd0feba4fc650e68ac8c19166ee1ba87f',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000_no_top.h5',
        'name': 'resnet50_imagenet_1000_no_top.h5',
        'md5': 'db3b217156506944570ac220086f09b6',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_places365_11586.h5',
        'name': 'resnet50_places365_11586.h5',
        'md5': 'bb8963db145bc9906452b3d9c9917275',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_11586_no_top.h5',
        'name': 'resnet50_imagenet_11586_no_top.h5',
        'md5': 'd8bf4e7ea082d9d43e37644da217324a',
    },

    # ResNet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000.h5',
        'name': 'resnet101_imagenet_1000.h5',
        'md5': '9489ed2d5d0037538134c880167622ad',
    },

    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000_no_top.h5',
        'name': 'resnet101_imagenet_1000_no_top.h5',
        'md5': '1016e7663980d5597a4e224d915c342d',
    },


    # ResNet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000.h5',
        'name': 'resnet152_imagenet_1000.h5',
        'md5': '1efffbcc0708fb0d46a9d096ae14f905',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000_no_top.h5',
        'name': 'resnet152_imagenet_1000_no_top.h5',
        'md5': '5867b94098df4640918941115db93734',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221.h5',
        'name': 'resnet152_imagenet11k_11221.h5',
        'md5': '24791790f6ef32f274430ce4a2ffee5d',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221_no_top.h5',
        'name': 'resnet152_imagenet11k_11221_no_top.h5',
        'md5': '25ab66dec217cb774a27d0f3659cafb3',
    },


    # ResNeXt50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000.h5',
        'name': 'resnext50_imagenet_1000.h5',
        'md5': '7c5c40381efb044a8dea5287ab2c83db',
    },

    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000_no_top.h5',
        'name': 'resnext50_imagenet_1000_no_top.h5',
        'md5': '7ade5c8aac9194af79b1724229bdaa50',
    },


    # ResNeXt101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000.h5',
        'name': 'resnext101_imagenet_1000.h5',
        'md5': '432536e85ee811568a0851c328182735',
    },

    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000_no_top.h5',
        'name': 'resnext101_imagenet_1000_no_top.h5',
        'md5': '91fe0126320e49f6ee607a0719828c7e',
    },

]
