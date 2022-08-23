from pickletools import optimize
from tkinter import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

# Assign labels to the data

training_dir = 'horse-or-human/training' # When running from the parent folder (i.e. humanhorses)
train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)
# Why rescale by 1./255?
# https://github.com/Arsey/keras-transfer-learning-for-oxford102/issues/1
# ur original images consist in RGB coefficients in the 0-255, but such values would be too high for our model to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

validation_dir = 'horse-or-human/validation'
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
  validation_dir,
  target_size=(300, 300),
  class_mode='binary'
)


# Training
model = tf.keras.models.Sequential([
    # 16 convolutions/filters, each a 3x3
    # activation is relu - function that just returns a value if it's greater than 0
    # 300x300 is the size of images in pixels and 3 is to represent color channels
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(300, 300, 3)),

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

# Training
history = model.fit(
  train_generator,
  epochs=15,
  validation_data=validation_generator
)

model.save('humanhorses')



# model.summary
# =================================================================
# conv2d (Conv2D)              (None, 298, 298, 16)  448
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 149, 149, 16)  0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 147, 147, 32)  4640
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 73, 73, 32)    0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 71, 71, 64)    18496
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 35, 35, 64)    0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 33, 33, 64)    36928
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)    0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 14, 14, 64)    36928
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)      0
# _________________________________________________________________
# flatten (Flatten)            (None, 3136)          0
# _________________________________________________________________
# dense (Dense)                (None, 512)           1606144
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)             513
# =================================================================
# Total params: 1,704,097
# Trainable params: 1,704,097
# Non-trainable params: 0
# _________________________________________________________________
# Note how, by the time the data has gone through all the convolutional and pooling layers, it ends up as 7 × 7 items. The theory is that these will be activated feature maps that are relatively simple, containing just 49 pixels. These feature maps can then be passed to the dense neural network to match them to the appropriate labels.
