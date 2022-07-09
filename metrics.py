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