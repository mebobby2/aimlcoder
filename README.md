# AI & ML for Coders

## To Run
* conda activate tf
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

## Terminology
* Hidden layers - layers that are between the inputs and the outputs aren't seen by a caller, so the term 'hidden' is used to describe them
* More neurons could also lead to a network that is great at recognizing the training data, but not so good at recognizing data that it hasn't previously seen (this is known as overfitting)
* In machine learning, a hyperparameter is a value that is used to control the training, as opposed to the internal values of the neurons that get trained/learned, which are referred to as parameters.
* In a scenario such as this one, the computer has no idea what the relationship between X and Y is. So it will make a guess. Say for example it guesses that Y = 10X + 10. It then needs to measure how good or how bad that guess is. That’s the job of the loss function.
* Armed with this knowledge, the computer can then make another guess. That’s the job of the optimizer. This is where the heavy calculus is used, but with TensorFlow, that can be hidden from you. You just pick the appropriate optimizer to use for differ‐ ent scenarios. In this case we picked one called sgd, which stands for stochastic gradi‐ ent descent—a complex mathematical function that, when given the values, the previous guess, and the results of calculating the errors (or loss) on that guess, can then generate another one. Over time, its job is to minimize the loss, and by so doing bring the guessed formula closer and closer to the correct answer.

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

## Upto
Page 57

Chapter 3
