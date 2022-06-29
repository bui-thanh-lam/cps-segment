from torchvision import transforms as T
import numpy as np


# pred: shape h * w
# target: shape h * w


def compute_iou(outputs, labels):
    SMOOTH = 1e-6
    outputs = outputs.view(-1)
    labels = labels.view(-1)
    intersection = (outputs * labels).sum()
    union = (outputs + labels).sum() - intersection
    iou = (intersection + SMOOTH) / (union + SMOOTH) 
    return iou.item()


def compute_dice(outputs, labels):
    SMOOTH = 1e-6
    outputs = outputs.view(-1)
    labels = labels.view(-1)
    intersection = (outputs * labels).sum()
    dice = (2.*intersection + SMOOTH) / (outputs.sum() + labels.sum() + SMOOTH)  
    return dice.item()


if __name__ == "__main__":
    import torch
    t_1 = torch.zeros((216, 216), dtype=torch.int)
    # t_2 = torch.zeros((216, 216), dtype=torch.int)
    t_2 = torch.randint(low=0, high=2, size=(216, 216))
    print(compute_dice(t_1, t_2))