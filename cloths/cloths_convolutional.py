import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# In this case, we want the layer to learn 64 convolutions. It will randomly initialize these, and over time will learn the filter values that work best to match the input val‐ ues to their labels. The (3, 3) indicates the size of the filter. Earlier I showed 3 × 3 filters, and that's what we are specifying here. This is the most common size of filter; you can change it as you see fit, but you'll typically see an odd number of axes like 5 × 5 or 7 × 7 because of how filters remove pixels from the borders of the image, as you'll see later.

# The activation and input_shape parameters are the same as before. As we're using Fashion MNIST in this example, the shape is still 28 × 28. Do note, however, that because Conv2D layers are designed for multicolor images, we're specifying the third dimension as 1, so our input shape is 28 × 28 × 1. Color images will typically have a 3 as the third parameter as they are stored as values of R, G, and B.

# Here's how to use a pooling layer in the neural network. You'll typically do this imme‐ diately after the convolutional layer. We split the image into 2 × 2 pools and picked the maximum value in each. This operation could have been parameterized to define the pool size. Those are the parameters that you can see here—the (2, 2) indicates that our pools are 2 × 2.


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'
),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=50)
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
