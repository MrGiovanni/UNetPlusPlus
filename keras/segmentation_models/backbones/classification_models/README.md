# Classification models Zoo
Pretrained classification models for Keras

### Models: 
- [ResNet](https://arxiv.org/abs/1512.03385) models converted from MXNet:
  - [ResNet18](https://github.com/qubvel/classification_models/blob/master/imgs/graphs/resnet18.png)
  - [ResNet34](https://github.com/qubvel/classification_models/blob/master/imgs/graphs/resnet34.png)
  - [ResNet50](https://github.com/qubvel/classification_models/blob/master/imgs/graphs/resnet50.png)
  - ResNet101
  - ResNet152
- [ResNeXt](https://arxiv.org/abs/1611.05431) models converted from MXNet:
  - ResNeXt50
  - ResNeXt101
  
| Model     | Classes |      Weights       | No top | Preprocessing|
|-----------|:-------:|:----------------------------:|:------:|:------:|
| ResNet18  | 1000  | `imagenet` | +  |BGR|
| ResNet34  | 1000  | `imagenet` | +  |BGR|
| ResNet50  | 1000<br>11586  |`imagenet`<br>`imagenet11k-place365ch` | +  |BGR |
| ResNet101 | 1000  | `imagenet` | +  |BGR |
| ResNet152 | 1000<br>11221| `imagenet`<br>`imagenet11k`| +  |BGR |
| ResNeXt50 | 1000 | `imagenet` | +  |- |
| ResNeXt101 | 1000 | `imagenet` | +  |- |


### Example  

Imagenet inference example:  
```python
import numpy as np
from skimage.io import imread
from keras.applications.imagenet_utils import decode_predictions

from classification_models import ResNet18
from classification_models.resnet import preprocess_input

# read and prepare image
x = imread('./imgs/tests/seagull.jpg')
x = preprocess_input(x, size=(224,224))
x = np.expand_dims(x, 0)

# load model
model = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=1000)

# processing image
y = model.predict(x)

# result
print(decode_predictions(y))
```

Model fine-tuning example:
```python
import keras
from classification_models import ResNet18

# prepare your data
X = ...
y = ...

n_classes = 10

# build model
base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.AveragePooling2D((7,7))(base_model.output)
x = keras.layers.Dropout(0.3)(x)
output = keras.layers.Dense(n_classes)(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# train
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y)
```
