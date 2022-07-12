import torch
from torchvision import transforms as T
from PIL import Image
import os
import random
import shutil
import numpy as np


IMAGE_SIZE = 340
DEVICE = torch.device('cuda')
IGNORE_INDEX = -1
TRAIN_INPUT_TRANSFORMS = T.Compose([
    T.ColorJitter(
        brightness=0.2,
        contrast=0.1,
        hue=0
    ),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])
TRAIN_SHARED_TRANSFORMS = T.Compose([
    T.RandomResizedCrop(
        (IMAGE_SIZE, IMAGE_SIZE),
        scale=(0.5, 2)
    ),
    T.RandomHorizontalFlip(),
])
TRAIN_TARGET_TRANSFORMS = None
VAL_INPUT_TRANSFORMS = T.Compose([
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])
VAL_SHARED_TRANSFORMS = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])
VAL_TARGET_TRANSFORMS = None


def convert_black_and_white_to_binary_mask(mask_dir, out_dir):
    for filename in os.listdir(mask_dir):
        with Image.open(os.path.join(mask_dir, filename)) as img:
            img = np.asarray(img)
            if len(img.shape) == 3:
                img = img[:, :, 0] // 200
            else:
                img = img // 200

            img = Image.fromarray(img)
            img.save(os.path.join(out_dir, filename))


def convert_model_output_to_black_and_white_mask(mask, out_dir, mask_name):
    mask = mask.cpu().numpy()
    mask = mask * 255
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.save(os.path.join(out_dir, mask_name))


def split_data_into_ssl_strategy(image_dir, mask_dir, out_dir, split_ratio=0.25):
    os.makedirs(os.path.join(out_dir, 'labelled', 'image'))
    os.makedirs(os.path.join(out_dir, 'labelled', 'mask'))
    os.makedirs(os.path.join(out_dir, 'unlabelled', 'image'))
    image_names = os.listdir(image_dir)
    chosen_images = random.sample(image_names, int(len(image_names) * split_ratio))
    for image in image_names:
        if image in chosen_images:
            shutil.copy(os.path.join(image_dir, image), os.path.join(out_dir, 'labelled', 'image', image))
            shutil.copy(os.path.join(mask_dir, image), os.path.join(out_dir, 'labelled', 'mask', image))
        else:
            shutil.copy(os.path.join(image_dir, image), os.path.join(out_dir, 'unlabelled', 'image', image))


def visualize_pseudo_labels():
    pass


if __name__ == "__main__":
    split_data_into_ssl_strategy(
        image_dir="../datasets/TrainDataset/image",
        mask_dir="../datasets/TrainDataset/mask",
        out_dir="../datasets/SemiDataset50",
        split_ratio=0.5
    )
