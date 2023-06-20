import torch
import torch.nn as nn
from torch.nn import functional as F

class NST_Loss(nn.Module):
    def __init__(
        self,
        styles_features: list[list[torch.Tensor]],
        content_features: list[torch.Tensor],
    ):
        super().__init__()

        content_features = [layer.detach() for layer in content_features]
        styles_features = [
            [style_feature.detach() for style_feature in style_features]
            for style_features
            in styles_features
        ]
        self.content_loss_fxn = ContentLoss(content_features)
        self.style_loss_fxn = StyleLoss(styles_features)

        self.alpha = 0.01
        self.beta = 0.2

    def forward(
        self,
        generated_image_features: tuple[list, list]
    ):
        generated_styles_features, generated_content_features = generated_image_features

        content_loss = self.content_loss_fxn(generated_content_features)
        style_loss = self.style_loss_fxn(generated_styles_features)

        total_loss = self.alpha * content_loss + self.beta * style_loss

        return (
            total_loss,
            {
                "content_loss": content_loss,
                "content_loss_alpha": self.alpha * content_loss,
                "style_loss": style_loss,
                "style_loss_beta": self.beta * style_loss,
                "total_loss": total_loss,
            }
        )


class ContentLoss(nn.Module):
    def __init__(self, content_features: list[torch.Tensor]):
        super().__init__()
        self.original_content_features = [layer.detach() for layer in content_features]

        num_layers = len(self.original_content_features)
        self.layers_weights = [1 for _ in range(num_layers)]

        assert len(self.layers_weights) == num_layers, \
        f"len(self.layers_weights) = {len(self.layers_weights)}, num_layers = {num_layers}"

    def forward(self, generated_content_features: list[torch.Tensor]):
        loss = 0
        for i, (gen_feat, orig_feat) in enumerate(zip(generated_content_features, self.original_content_features)):
            loss += F.mse_loss(gen_feat, orig_feat) * self.layers_weights[i]

        return loss

class StyleLoss(nn.Module):
    def __init__(self, styles_features: list[list[torch.Tensor]]):
        super().__init__()
        self.original_styles_features = styles_features

        self.num_style_images = len(self.original_styles_features)
        self.num_layers = len(self.original_styles_features[0])

        self.image_weights = [1 for _ in range(self.num_style_images)]
        self.layers_weights = [1, 0.8, 0.6, 0.3, 0.1]

        assert len(self.image_weights) == self.num_style_images, \
        f"len(self.image_weights) = {len(self.image_weights)}, self.num_style_images = {self.num_style_images}"
        assert len(self.layers_weights) == self.num_layers, \
        f"len(self.layers_weights) = {len(self.layers_weights)}, self.num_layers = {self.num_layers}"

        # self.original_styles_features = [style_feature.detach() for style_feature in style_features]

    def forward(self, generated_style_features: list[torch.Tensor]):
        loss = 0
        for origi_style_features in self.original_styles_features:  # image Loop
            style_loss = 0
            for i, (gen_feat, orig_feat) in enumerate(zip(generated_style_features, origi_style_features)): # layer Loop
                gen_gram = self.gram_matrix(gen_feat)
                orig_gram = self.gram_matrix(orig_feat)
                _, c, h, w = gen_feat.size()
                style_loss += F.mse_loss(gen_gram, orig_gram) * self.layers_weights[i] / (c * h * w)

            loss += style_loss

        return loss

    def gram_matrix(self, x):
        _, c, h, w = x.size()
        x = x.view(c, h * w)
        G = torch.mm(x, x.t())

        return G