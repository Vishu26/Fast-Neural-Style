import torch.nn.functional as F
from torch import nn
from utils import imagenet_preprocess, gram_matrix


class PerceptualLoss(nn.Module):
    def __init__(self, loss_network, content_weight=1, style_weight=1e5):
        super(PerceptualLoss, self).__init__()
        self.loss_network = loss_network
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.style_criterion = StyleLoss(loss_network)
        self.content_criterion = ContentLoss(loss_network)

    def forward(self, output, original_img, style_image):
        content_loss = self.content_criterion(output, original_img) * self.content_weight
        style_loss = self.style_criterion(output, style_image) * self.style_weight
        return style_loss, content_loss


class StyleLoss(nn.Module):
    """
    Must be initialized with a "loss network"
    Example use  VGG relu2_2 as loss network:
    vgg = vgg16(pretrained=True)
    relu2_2 = nn.Sequential(*list(vgg.features)[:9])
    for param in relu2_2.parameters():
        param.requires_grad = False
    relu2_2.eval
    relu2_2.cuda()
    """
    def __init__(self, loss_network):
        super(StyleLoss, self).__init__()
        self.loss_network = loss_network

    def forward(self, output, style):
        loss = 0
        features_output = self.loss_network(output)
        features_style = self.loss_network(style)
        for layer in features_output.keys():
            loss += F.mse_loss(gram_matrix(features_output[layer]),
                               gram_matrix(features_style[layer]))
        return loss


class ContentLoss(nn.Module):
    """
    Must be initialized with a "loss network"
    Example use  VGG relu2_2 as loss network:
    vgg = vgg16(pretrained=True)
    relu2_2 = nn.Sequential(*list(vgg.features)[:9])
    for param in relu2_2.parameters():
        param.requires_grad = False
    relu2_2.eval
    relu2_2.cuda()
    """
    def __init__(self, loss_network):
        super(ContentLoss, self).__init__()
        self.loss_network = loss_network

    def forward(self, input, target):
        features_input = self.loss_network(input)
        features_target = self.loss_network(target)
        return F.mse_loss(features_input['relu3_3'],
                          features_target['relu3_3'])
