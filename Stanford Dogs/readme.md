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

## Image Classification Models
Since The dataset is large and different classes are quite similar. I have used `deeper models` and huge emphasis is given on `Transfer Learning`(TL).

Checkout [this notebook](<classification/Transfer-Learning-Pipeline(Fully Trained).ipynb>), it has all the code for training and testing the models.

### Results
The results are quite good. The best model is `InceptionResNetV2` with `84.4%` accuracy and `97.6%` top-5 accuracy. The model is trained for `35 epochs` with `batch size of 64` and `Adam` optimizer with `learning rate of 4e-4`.
| S. No | Model | Accuracy | Top-5 Accuracy |
|---|---|---|---|
| 1 | InceptionResNetV2 | **84.4%** | **97.6%** | 0 secs |
| 2 | DenseNet201       | **81.1%** | **96.8%** | 0 secs |
| 3 | InceptionV3       | **81.0%** | **97.1%** | 0 secs |
| 4 | Xception          | **80.6%** | **96.8%** | 0 secs |
| 5 | DenseNet169       | **79.5%** | **96.4%** | 0 secs |
| 6 | ResNet152V2       | **78.1%** | **96.5%** | 0 secs |
| 7 | MobileNet         | **76.7%** | **96.1%** | 0 secs |
| 8 | ResNet101V2       | **76.7%** | **96.1%** | 0 secs |
| 9 | DenseNet121       | **76.2%** | **95.9%** | 0 secs |
| 10| MobileNetV2       | **76.0%** | **95.2%** | 0 secs |
| 11| ResNet50V2        | **75.0%** | **95.7%** | 0 secs |
