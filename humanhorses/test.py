import numpy as np
import os
from  tensorflow import keras
from tensorflow.keras.preprocessing import image

model = keras.models.load_model('..')

for filename in os.listdir('testimages'):
  f = os.path.join('testimages', filename)

  # predicting images
  img = image.load_img(f, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  image_tensor = np.vstack([x])
  classes = model.predict(image_tensor)
  print(classes)
  print(classes[0])
  if classes[0]>0.5:
    print(filename + " is a human")
  else:
    print(filename + " is a horse")
