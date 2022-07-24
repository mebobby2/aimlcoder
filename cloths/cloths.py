import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

# The first, Flatten, isn't a layer of neurons, but an input layer specification. Our inputs are 28 × 28 images, but we want them to be treated as a series of numeric values
# The next one, Dense, is a layer of neurons, and we're specifying that we want 128 of them. This is the middle layer shown in Figure 2-5. You'll often hear such layers described as hidden layers. Layers that are between the inputs and the outputs aren't seen by a caller, so the term "hidden" is used to describe them. We're asking for 128 neurons to have their internal parameters randomly initialized. Often the question I'll get asked at this point is "Why 128?" This is entirely arbitrary—there's no fixed rule for the number of neurons to use. As you design the layers you want to pick the appropriate number of values to enable your model to actually learn. More neurons means it will run more slowly, as it has to learn more parameters. More neurons could also lead to a network that is great at recognizing the training data, but not so good at recognizing data that it hasn't previously seen (this is known as overfitting, and we'll discuss it later in this chapter). On the other hand, fewer neurons means that the model might not have sufficient parameters to learn.

# It takes some experimentation over time to pick the right values. This process is typically called hyperparameter tuning. In machine learning, a hyperparameter is a value that is used to control the training, as opposed to the internal values of the neurons that get trained/learned, which are referred to as parameters.

# You might notice that there's also an activation function specified in that layer. The activation function is code that will execute on each neuron in the layer. TensorFlow supports a number of them, but a very common one in middle layers is relu, which stands for rectified linear unit. It's a simple function that just returns a value if it's greater than 0. In this case, we don't want negative values being passed to the next layer to potentially impact the summing function, so instead of writing a lot of if-then code, we can simply activate the layer with relu.

# Finally, there's another Dense layer, which is the output layer. This has 10 neurons, because we have 10 classes. Each of these neurons will end up with a probability that the input pixels match that class, so our job is to determine which one has the highest value. We could loop through them to pick that value, but the softmax activation function does that for us.

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = myCallback()
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

# Label	Description
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

test_image_index = 1155
print(classifications[test_image_index])
print(test_labels[test_image_index])

plt.imshow(test_images[test_image_index], cmap='gray')
plt.show()
