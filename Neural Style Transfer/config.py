import torch

device=torch.device("mps" if torch.cuda.is_available() else "cpu")
STYLE_IMAGES_PATH="./style images/"

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_feature_layers=[0, 5, 10, 19, 28]
content_feature_layers=[21]
alpha=1
beta=1
learning_rate=0.1
epochs=1000
