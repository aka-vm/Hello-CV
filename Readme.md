# Hello CNN
Not just a **hello-world Repository** with entery-level code!<br>
In this repo I have worked to develop various models for the purpose of Image Classification and used them on various datasets.<br>
I have used **Keras** and **TensorFlow** as my backend framework.<br>

- [Hello CNN](#hello-cnn)
  - [TODOs for running the code](#todos-for-running-the-code)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Results](#results)

## TODOs for running the code
1. Create `paths.py` file (use `paths.py.template` as a template)
2. Download the datasets.

## Datasets
All the used datasets well-known around the deeplearning community. They include -
* [MNIST - Digit Recognizer](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data): A dataset of handwritten digits.
* [CIFAR-10](https://www.kaggle.com/competitions/cifar-10/data): A dataset of 32x32 color images of 10 classes of objects.

   For Larger I have used images and directories instead of single objects. This way I can use directory iterator to load images, which uses less memory.

## Models
I have used various models to classify the images, they vary from simple **Sequential Models** to complex **Functional Models**, **Transfer Learning Models** and more. All the models that are integrated on the pipeline include -

* **Simple Sequential Model**: A simple model that is based on [keras.models.Sequential](https://keras.io/models/sequential/) class.

* **Deep Sequential Model**: A model that is based on [keras.models.Sequential](https://keras.io/models/sequential/) class, better than **Simple Sequential Model**.

* **VGG-Like Model**: This model contains blocks similar to **VGG-16**.

* **Resnet-Like Model**: This model contains **Redidual Blocks**, similar to **Resnet Models**.

## Results
| S. No | Dataset | Best Accuracy | Best Model |
|---|---|---|---|
| 1 | MNIST(Digit Recognizer) | 99.3% | Simple Sequential |
| 2 | CIFAR-10 | 84.1% | VGG-Like Model |



<hr>



Check the branches. I have created a new branch everytime I added a new type of model.<br>
NOTE: For Some reason different hardware give different results, I used two Machines, **Macbook Air M1(8gb)** and **Intel i7 11700k | RTX3070**. **RTX3070** machine gave better results with a good margin.