import torch.nn as nn
from torchvision import models

def create_model(num_classes):

    model = models.squeezenet1_0(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(
            512,
            num_classes,
            kernel_size=1
        ),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1,1))
    )

    model.num_classes = num_classes

    return model
