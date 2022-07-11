# Stanford Dogs

## Dataset

The [Stanford Dogs dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) contains images of 120 breeds of dogs. This dataset contains uneven images with different ratio and pixel-size.

#### Image Data Specification
| Data Spcifications | Value |
|---|---|
| Total Classes | 120 |
| Total Images | 20,580 |
| Pixel Size | Different |
| Pixel Ratio | Different |

#### Data Distribution
| Set | Ratio | Number |
|---|---|---|
| Train | 0.75 | 15394 |
| Test | 0.25 | 5,186 |


## Image Preprocessing and Augmentation

## Models
Since The dataset is large and different classes are quite similar. I have used `deeper models` and huge emphasis is given on `Transfer Learning`(TL).

A new type of learning process is also used, I've named it `Partial Transfer Learning`(PTL), In this the TL counterpart is imported and we make a certian part of CNN trainable, this is selected based on probability.

* **InceptionNet-V3**: This model is based on [InceptionNet-V3](https://keras.io/api/applications/inceptionv3/). So far, This model is the best performing model in the dataset. The take away from this model is that using inseption layers is a good idea.

* **ResNet152-V2**: This model is based on [ResNet152-V2](https://keras.io/api/applications/resnet/#resnet152v2-function). It has given a respectable performance. The take away from this model is that taking residual blocks will also work great.

* **VGG-19**: This model is based on [VGG-19](https://keras.io/api/applications/vgg/#vgg19-function). So far, This model has given the worst performance. The take away from this model is that for good results we need to put emphasis on residual and inception layers, Using only deep layers is not a good idea.



## Results
| S. No | Model | Accuracy | Top-5 Accuracy |
|---|---|---|---|
| 1 | [InceptionNet-V3(PTL)](/Stanford%20Dogs/PTL-Inception-net-V3.ipynb) | **81.51%** | **96.93%** | 0 secs |
| 2 | [InceptionNet-V3(TL)](/Stanford%20Dogs/TL-Inception-net-V3.ipynb) | **80.81%** | **96.91%** | 0 secs |
| 3 | [Xception(TL)](/Stanford%20Dogs/TL-Xception.ipynb) | **80.81%** | **96.51%** | 0 secs |
| 4 | [ResNet152-V2(TL)](/Stanford%20Dogs/TL-ResNet.ipynb) | **78.71%** | **95.99%** | 0 secs |
| 5 | [VGG-19(TL)](/Stanford%20Dogs/TL-VGG.ipynb) | **25.99%** | **55.36%** | 0 secs |
<!-- | n | [Model](/Stanford%20Dogs/#) | **00%** | **00%** | 0 secs | -->
̀