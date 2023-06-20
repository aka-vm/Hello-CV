import torch
import torch.nn as nn
from torchvision import models

class Network(nn.Module):
    def __init__(
        self,
        style_images: list[torch.Tensor],
        content_image: torch.Tensor = None,
    ):
        super().__init__()
        device = content_image.device
        self.style_feature_layers = [0, 5, 10, 19, 28]
        self.content_feature_layers = [21, 30]
        max_layer = max(self.style_feature_layers + self.style_feature_layers)
        self.vgg_backbone = models.vgg19(pretrained=True).features[:max_layer + 1].to(device).eval()


        self.style_images = style_images
        self.content_image = content_image
        self.set_style_images(style_images)
        if content_image is not None:
            self.set_content_image(content_image)


    def set_style_images(self, style_images: list):
        self.style_features = []
        for style_image in style_images:
            self.style_features.append(self.forward(style_image)[0])
        self.num_style_images = len(style_images)

    def set_content_image(self, content_image):
        self.content_features = self.forward(content_image)[1]

    def forward(self, x):
        style_features = []
        content_features = []
        for i, layer in enumerate(self.vgg_backbone):
            x = layer(x)
            if i in self.style_feature_layers:
                style_features.append(x)
            if i in self.content_feature_layers:
                content_features.append(x)

        return style_features, content_features
