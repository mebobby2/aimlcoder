# AI & ML for Coders

## To Run
* python3 script.py

## Install Tensorflow
### Install Miniconda
* curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
* bash Miniconda3-latest-MacOSX-x86_64.sh
* You may need to restart your terminal

### Create a conda environment
* conda create --name tf python=3.9
* conda activate tf
  * conda deactivate

### Install Tensorflow
* pip install --upgrade pip
* pip install tensorflow

### Verify
* python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

Instructions: https://www.tensorflow.org/install/pip#macos

### USE THESE INSTRUCTIONS ON APPLE M1 CHIPS
* https://developer.apple.com/forums/thread/686926
* https://developer.apple.com/metal/tensorflow-plugin/

* python3 -m pip install tensorflow-macos
  * This is installing tensorflow onto the system - not a virtual env through Conda

## Install Other Dependencies
* python3 -m pip install matplotlib

## Terminology
* Hidden layers - layers that are between the inputs and the outputs aren't seen by a caller, so the term 'hidden' is used to describe them
* More neurons could also lead to a network that is great at recognizing the training data, but not so good at recognizing data that it hasn't previously seen (this is known as overfitting)
* In machine learning, a hyperparameter is a value that is used to control the training, as opposed to the internal values of the neurons that get trained/learned, which are referred to as parameters.
* In a scenario such as this one, the computer has no idea what the relationship between X and Y is. So it will make a guess. Say for example it guesses that Y = 10X + 10. It then needs to measure how good or how bad that guess is. That's the job of the loss function.
* Armed with this knowledge, the computer can then make another guess. That's the job of the optimizer. This is where the heavy calculus is used, but with TensorFlow, that can be hidden from you. You just pick the appropriate optimizer to use for differ‐ ent scenarios. In this case we picked one called sgd, which stands for stochastic gradi‐ ent descent—a complex mathematical function that, when given the values, the previous guess, and the results of calculating the errors (or loss) on that guess, can then generate another one. Over time, its job is to minimize the loss, and by so doing bring the guessed formula closer and closer to the correct answer.
* What is a Dense Layer in Neural Network? Is a neural network layer that is connected deeply, which means each neuron in the dense layer receives input from all neurons of its previous layer. The dense layer is found to be the most commonly used layer in the models.

## ML Models
### Convolutional Neural Network (CNN)
#### Convolutions
Convolutions - A convolution is simply a filter of weights that are used to multiply a pixel with its neighbors to get a new value for the pixel.

If we then define a filter in the same 3 × 3 grid, as shown below the original values, we can transform that pixel by calculating a new value for it. We do this by multiplying the current value of each pixel in the grid by the value in the same position in the filter grid, and summing up the total amount. This total will be the new value for the current pixel. We then repeat this for all pixels in the image.

Repeating this process across every pixel in the image will give us a filtered image.

Let's consider the impact of applying a filter on a more complicated image.

Using a filter with negative values on the left, positive values on the right, and zeros in the middle will end up removing most of the information from the image except for vertical lines. Similarly, a small change to the filter can emphasize the horizontal lines.

These examples also show that the amount of information in the image is reduced, so we can potentially learn a set of filters that reduce the image to features, and those features can be matched to labels as before. Previously, we learned parameters that were used in neurons to match inputs to outputs. Similarly, the best filters to match inputs to outputs can be learned over time.

When combined with pooling, we can reduce the amount of information in the image while maintaining the features. We'll explore that next.

#### Pooling
Pooling is the process of eliminating pixels in your image while maintaining the semantics of the content within the image.

Consider the box on the left to be the pixels in a monochrome image. We then group them into 2 × 2 arrays, so in this case the 16 pixels are grouped into four 2 × 2 arrays. These are called pools.

We then select the maximum value in each of the groups, and reassemble those into a new image. Thus, the pixels on the left are reduced by 75% (from 16 to 4), with the maximum value from each pool making up the new image.

Note how the filtered features have not just been maintained, but further enhanced. Also, the image size has changed from 512 × 512 to 256 × 256—a quarter of the original size.

There are other approaches to pooling, such as min pooling, which takes the smallest pixel value from the pool, and average pooling, which takes the overall average value.

#### Training, Validation, and Testing
You may be wondering why we’re talking about a validation dataset here, rather than a test dataset, and whether they’re the same thing. For simple models like the ones developed in the previous chapters, it’s often sufficient to split the dataset into two parts, one for train‐ ing and one for testing. But for more complex models like the one we’re building here, you’ll want to create separate validation and test sets. What’s the difference? Training data is the data that is used to teach the network how the data and labels fit together. Validation data is used to see how the network is doing with previously unseen data while you are training—i.e., it isn’t used for fitting data to labels, but to inspect how well the fitting is going. Test data is used after training to see how the network does with data it has never previously seen. Some datasets come with a three-way split, and in other cases you’ll want to separate the test set into two parts for validation and testing. Here, you’ll download some additional images for testing the model.


## Python
### Wheels
https://realpython.com/python-wheels/#:~:text=and%20its%20dependencies.-,What%20Is%20a%20Python%20Wheel%3F,a%20type%20of%20built%20distribution.

What Is a Python Wheel?
A Python .whl file is essentially a ZIP (.zip) archive with a specially crafted filename that tells installers what Python versions and platforms the wheel will support.

A wheel is a type of built distribution. In this case, built means that the wheel comes in a ready-to-install format and allows you to skip the build stage required with source distributions.

### Conda vs Pip
https://www.anaconda.com/blog/understanding-conda-and-pip

Conda - manages binaries, has the ability to create virtual environments, does dependency checks
Pip - manages wheel or python source code only, does not create virtual environments OOTB but can be done using virtualenv or venv, does not do dependency checks

## Good Online Sources
https://github.com/christianversloot/machine-learning-articles

## Book Source Code
https://github.com/lmoroney/tfbook

## Upto
Page 82

The results here are much better than with our previous mode

Before that:
Figure out why this model predicts all my test images (including some images I took from the validation set) as human!

Try these suggestions on improving the accurracy of the dog or cat model. Forget about the human or horse model as I only have a small set of data to play with anyways
https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class
https://theorangeduck.com/page/neural-network-not-working#preprocess
