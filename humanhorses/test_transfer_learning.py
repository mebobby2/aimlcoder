import numpy as np
import os
from  tensorflow import keras
from tensorflow.keras.preprocessing import image

model = keras.models.load_model('transferred_model')

for filename in os.listdir('testimages'):
  f = os.path.join('testimages', filename)

  # predicting images
  img = image.load_img(f, target_size=(150, 150)) # Resize images to 300x300, as that's what our model is trained on
  x = image.img_to_array(img) # Converts image to a 2D array
  x = np.expand_dims(x, axis=0) # Our model is trained on 3D arrays, hence, we convert the 2D to a 3D array

  image_tensor = np.vstack([x])
  classes = model.predict(image_tensor)
  print(classes)
  print(classes[0])
  if classes[0]>0.5:
    print(filename + " is a human")
  else:
    print(filename + " is a horse")
