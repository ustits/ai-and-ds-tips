"""
documentation = https://keras.io/applications/

Instead of modeling we can use pretrained models
Downloads weights to ~/.keras/models so will take some time to evaluate
"""

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

inception = InceptionV3(weights='imagenet')
inception.summary()

img_path = '../open_cv_tips/images/cat_example.jpg'

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = inception.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
