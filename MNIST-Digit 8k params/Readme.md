# MNIST Digit Classification with 8k parameters

In this project, I'm going to document my journey of creating a model that can classify the MNIST dataset with the following conditions-
* 25 Epochs
* 8k Parameters
* 99%+ Validation Accuracy


## Why the Hell?
First of all, While doing this challenge I'm Jobless, It was suggested by a senior he emphasized on the fact that It'll be a good learning experience.

One Good Application of this is that we can use this model in a microcontroller for inference. 8k parameters means  memory of 16kB(Float16), For an ESP-MCU this amount of memory is available. So, we can use this model in a microcontroller.

I'll be using Pytorch for this project.

## Iteration 1: Simple Fully Connected Network

So, Lets make things clear 8k is not a lot of parameters. So, we need to make sure that we use them wisely. The image size is `28x28` pixels. We can't use a lot of convolutions.
Here are the things I'm going to try-

* 3x3 Convolutions
* 4 Convolutions
* filters = `8, 8, 16, 32`
* 2 max pooling layers
* No Padding
* ReLU Activation

This will give us a total of `7,762` trainable parameters.
