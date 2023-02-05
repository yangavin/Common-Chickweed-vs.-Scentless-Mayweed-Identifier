from keras.models import load_model
from keras.utils import image_utils
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
 
from keras.models import load_model
 
model = load_model('model_saved.h5')

image = image_utils.load_img('Example Chickweed.jpeg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print("Predicted Class (0 - Common Chickweed , 1 - Scentless Mayweed): ", label[0][0])