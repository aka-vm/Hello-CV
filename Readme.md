# Hello CV
`Not just a hello-world Repository with entery-level code!`


In this repo I have worked to develop various Computer Vision taskes. The code uses both Pytorch & Tensorflow.

## TODOs for running the code
1. Create `paths.py` file (use `paths.py.template` as a template)
2. Download the datasets.

## Datasets
All the used datasets well-known around the deeplearning community. They include -
* [MNIST - Digit Recognizer](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data): A dataset of handwritten digits.
* [CIFAR-10](https://www.kaggle.com/competitions/cifar-10/data): This dataset contains 32x32 color images of 10 classes of objects.
* [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset): Contains images of 120 breeds of dogs. Each image has different pixel values and ratio.

### Data Preprocessing

* For large datasets I have used `Image Iterator`, This uses less memory.

* Image augmentation is also used to augment the images.

## Image Classification Results
| S. No | Dataset | Best Accuracy | Best Model | Real-Life Test |
|---|---|---|---|---|
| 1 | [MNIST(Digit Recognizer)](/MNIST-Digit_Recogonizer/) | 99.3% | Simple Sequential CNN|
| 2 | [CIFAR-10](/CIFAR-10/) | **89.94%** | [VGG-Like Model](/CIFAR-10/final-notebook.ipynb) |  [5 images](/CIFAR-10/real-image-test.ipynb) |
| 3 | [Stanford Dogs](/Stanford%20Dogs/) | **84.3%** | [InceptionResNetV2](<Stanford Dogs/classification/Transfer-Learning-Pipeline(Fully Trained).ipynb>) | [4 images](/Stanford%20Dogs/real-image-test.ipynb) |

## Other Computer Vision Tasks
| S. No | Application Name  |Refered Literature/Implimentation| Implimented Using | Metric | Score | Visuals |
| ---| ---| ---| ---| ---| ---| ---|
| 1 | [Neural Style Transfer](/Neural%20Style%20Transfer/)|[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576),<br>[NST With Two Style](https://towardsdatascience.com/mixed-neural-style-transfer-with-two-style-images-9469b2681b54) | Pytorch |  |  | ![gif](<Neural Style Transfer/car-van-g.gif>) |
| 2 | [Dog Breed Detection (YOLOv8)](/Stanford%20Dogs/detection/yolo-v8.ipynb)|[Joseph Chet's Publications](https://pjreddie.com/publications/), [YOLOv8 Implimentation](https://github.com/ultralytics/ultralytics)| Pytorch, Ultralytics | mAP50-95 | 0.79 | ![Alt text](<Stanford Dogs/detection/images/dog mother.gif>) |
<!-- | 2 | [Face ](/Neural%20Style%20Transfer/) | [A Neural Algorithm  of<br> Artistic Style](https://arxiv.org/abs/1508.06576) | Pytorch | -->

<hr><br><br>

`Check the branches`. I have created a new branch everytime I added a new type of model.<br><br>
NOTE: For Some reason different hardware give different results, I used two Machines, **Macbook Air `M1(8gb)`** and **Intel i7 11700k | `RTX3070`**. **`RTX3070`** machine gave better results with a good margin. I even used [**Kaggle**](https://www.kaggle.com/) and [**Jarvis Labs**](https://jarvislabs.ai)to train some of the models.
