import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Consider these two sets of numbers:
# X=–1,0,1,2,3,4
# Y=–3,–1,1,3,5,7
# There's a relationship between the X and Y values (for example, if X is –1 then Y is –3, if X is 3 then Y is 5, and so on). Can you see it?

# After a few seconds you probably saw that the pattern here is Y = 2X – 1

#  There's only one layer, and it contains only one neuron

# “Dense” means a set of fully (or densely) connected neurons, which is what you can see in Figure 1-18 where every neuron is connected to every neuron in the next layer. It’s the most common form of layer type. Our Dense layer has units=1 specified, so we have just one dense layer with one neuron in our entire neural network. Finally, when you specify the first layer in a neural network (in this case, it’s our only layer), you have to tell it what the shape of the input data is. In this case our input data is our X, which is just a single value, so we specify that that’s its shape.

# In a scenario such as this one, the computer has no idea what the relationship between X and Y is. So it will make a guess. Say for example it guesses that Y = 10X + 10. It then needs to measure how good or how bad that guess is. That’s the job of the loss function.

# It already knows the answers when X is –1, 0, 1, 2, 3, and 4, so the loss function can compare these to the answers for the guessed relationship. If it guessed Y = 10X + 10, then when X is –1, Y will be 0. The correct answer there was –3, so it’s a bit off. But when X is 4, the guessed answer is 50, whereas the correct one is 7. That’s really far off.

# Armed with this knowledge, the computer can then make another guess. That’s the job of the optimizer. This is where the heavy calculus is used, but with TensorFlow, that can be hidden from you. You just pick the appropriate optimizer to use for different scenarios. In this case we picked one called sgd, which stands for stochastic gradient descent—a complex mathematical function that, when given the values, the previous guess, and the results of calculating the errors (or loss) on that guess, can then generate another one. Over time, its job is to minimize the loss, and by so doing bring the guessed formula closer and closer to the correct answer.

# The model only has a single neuron in it, and that neuron learns a weight and a bias, so that Y = WX + B. This looks exactly like the relationship Y = 2X – 1 that we want, where we would want it to learn that W = 2 and B = –1. Given that the model was trained on only six items of data, the answer could never be expected to be exactly these values, but something very close to them.

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
print("here is what I learned: {}".format(l0.get_weights()))
