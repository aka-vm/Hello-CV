import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms

import numpy as np
from PIL import Image

from network import Network
import config


tensor_loader = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    ),
])
tensor_unloader = transforms.Compose([
    transforms.Normalize(
        (-2.12, -2.04, -1.80),
        (4.37, 4.46, 4.44)
    ),
    transforms.ToPILImage()
])

def load_image(image_path: str, ):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path)

    img_shape = image.size
    scale_factor = 400 / max(img_shape)
    new_img_shape = np.array(img_shape) * scale_factor
    image = image.resize(new_img_shape.astype(int), Image.Resampling.LANCZOS)

    image = tensor_loader(image).unsqueeze(0)
    return image.to(config.device)

def main():
    style_images = [
        load_image(path_)
        for path_
        in config.STYLE_IMAGES_PATH
    ]

    content_image = load_image("/Users/vineetmahajan/Code/AI/Projects/Hello/Hello-CNN/Neural Style Transfer/content images/The-Guardian.jpg")

    NST_Network = Network(style_images, content_image)

if __name__ == '__main__':
    main()