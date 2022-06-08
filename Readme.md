# Hello CNN
Not just a **hello-world Repository** with entery-level code!<br>
In this repo I have worked to develop various models for the purpose of Image Classification and used them on various datasets.<br>
I have used **Keras** and **TensorFlow** as my backend framework.<br>

## TODOs for running the code
1. Create `paths.py` file (use `paths.py.template` as a template)
2. Download the datasets.

## Datasets
All the used datasets well-known around the deeplearning community. They include -
* [MNIST - Digit Recognizer](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data): A dataset of handwritten digits.
<!-- * [CIFAR-10](https://www.kaggle.com/competitions/cifar-10/data): A dataset of 32x32 color images of 10 classes of objects. -->

   For Larger I have used images and directories instead of single objects. This way I can use directory iterator to load images, which uses less memory.

## Models
I have used various models to classify the images, they vary from simple **Sequential Models** to complex **Functional Models**, **Transfer Learning Models** and more. All the models that are integrated on the pipeline include -

* **Simple Sequential Model**: A simple model that is based on [keras.models.Sequential](https://keras.io/models/sequential/) class.


<hr>



Check the branches. I have created a new branch everytime I added a new type of model.