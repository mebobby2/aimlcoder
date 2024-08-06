# Install TensorFlow on Apple M1
https://stackoverflow.com/questions/72964800/what-is-the-proper-way-to-install-tensorflow-on-apple-m1-in-2022

First of all, TensorFlow does not support officially the Mac M1. They don't distribute packages precompiled for the Mac M1 (and its specific arm64 arch), hence the tensorflow-macos package, which is maintained by Apple. TensorFlow distributes, as far as I know, official wheels only for x86 (Linux, Windows, Mac), and the Raspberry PI (arm64).

Apple is using a specific plugin in Tensorflow to make the framework compatible with Metal, the graphic stack of MacOS. To put it in a other way, they are leveraging the PluggableDevice API of Tensorflow to write code that translates the TensorFlow operations to code that the GPU of the M1 understands.

Those two packages contain respectively:

* tensorflow-deps the dependencies to run Tensorflow on arm64, i.e python, numpy, grpcio and h5py. This is more of a convenience package, I believe.
* tensorflow-metal: a plugin to make tensorflow able to run on metal, the shader API of MacOS (comparable to the low level APIs of Vulkan or DirectX12 on other platforms). You can think of it as a replacement of CUDA, if you are used to run TensorFlow on Nvidia GPUs.
Without the tensorflow-metal package, TensorFlow won't be able to leverage the GPU of the M1, but will still be able to run code on the CPU.

## Instructions
(Adapted from https://stackoverflow.com/questions/72964800/what-is-the-proper-way-to-install-tensorflow-on-apple-m1-in-2022)

Find base architecture of computer
```
conda config --show subdir
```

For **Native (osx-arm64) base**
```
conda env create -n my_tf_env -f tf-metal-arm64.yaml
```

For **Emulated (osx-64) base**
```
CONDA_SUBDIR=osx-arm64 mamba env create -n my_tf_env -f tf-metal-arm64.yaml
```

Verify TF installed
```
conda activate my_tf_env

python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Install other useful libraries
```
conda activate my_tf_env

pip install pillow
```

Issues with tensorflow-metal (https://github.com/tensorflow/tensorflow/issues/64507)
```
I understand that you are using GPU to build your model. This is an issue with the tensorflow-metal library, as it only supports tensorflow up to version 2.14. Therefore, it seems that we need to wait for a new version of the tensorflow-metal library. For now, if you want to use tensorFlow >2.14, you will need to uninstall tensorflow-metal or refrain from using the GPU.
```

So, unintall tensorflow-metal for now
```
pip uninstall tensorflow-metal
```
