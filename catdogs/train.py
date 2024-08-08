from pickletools import optimize
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tkinter import Image
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

TRAINING_DIR = "/Users/BobbyLei/Desktop/learn/aimlcoder/catdogs/cat-or-dog/training/"
# Experiment with your own parameters here to really try to drive it to 99.9% accuracy or better
train_datagen = ImageDataGenerator(rescale=1./255,
    #   rotation_range=40,
    #   width_shift_range=0.2,
    #   height_shift_range=0.2,
    #   shear_range=0.2,
    #   zoom_range=0.2,
    #   horizontal_flip=True,
    #   fill_mode='nearest'
      )
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "/Users/BobbyLei/Desktop/learn/aimlcoder/catdogs/cat-or-dog/testing/"
# Experiment with your own parameters here to really try to drive it to 99.9% accuracy or better
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

model = tf.keras.models.Sequential([
    # 16 convolutions/filters, each a 3x3
    # activation is relu - function that just returns a value if it's greater than 0
    # 300x300 is the size of images in pixels and 3 is to represent color channels
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),

    # Split image into 2 x 2 pools and pick max value in each (this eliminates pixels while also enhances the semantics)
    tf.keras.layers.MaxPooling2D(2, 2),

    # Stack several convolutional layers.
    # We do this because our image source is quite large, and we want, over time, to have many smaller images, each with features highlighted
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the images
    tf.keras.layers.Flatten(),

    # A layer of 512 neurons
    tf.keras.layers.Dense(512, activation='relu'),

    # A layer of 1 neurons (this is binary classifier)
    # Sigmoid function drives one set of values toward 0 and the other toward 1, which is perfect for binary classification.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
  optimizer=RMSprop(learning_rate=0.001),
  metrics=['accuracy'])


history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=20,
            verbose=1)

model.save('catordog.keras')
