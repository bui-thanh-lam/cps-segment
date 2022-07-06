from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from models import resnet
from models.utils import IntermediateLayerGetter, _SimpleSegmentationModel, _load_weights
from models.fcn import FCNHead

__all__ = [
    "DeepLabV3",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_resnet34",
]

model_urls = {
    "deeplabv3_resnet50_coco": "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_resnet101_coco": "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
    "deeplabv3_mobilenet_v3_large_coco": "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
}


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)

        return self.project(res)


def _deeplabv3_resnet(
        backbone: resnet.ResNet,
        num_classes: int,
        aux: Optional[bool],
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(256, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def deeplabv3_resnet50(
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 2,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if pretrained:
        arch = "deeplabv3_resnet50_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def deeplabv3_resnet101(
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 2,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool, optional): If True, include an auxiliary classifier
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet101(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if pretrained:
        arch = "deeplabv3_resnet101_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def deeplabv3_resnet34(
        pretrained: bool = False,
        num_classes: int = 2,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool, optional): If True, include an auxiliary classifier
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = True

    backbone = resnet.resnet34(pretrained=pretrained_backbone)
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)
    model.classifier = DeepLabHead(in_channels=512, num_classes=2)
    return model
