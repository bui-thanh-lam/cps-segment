import torch
from torchvision import transforms as T
from PIL import Image
import os
import random
import shutil
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
DEVICE = torch.device('cuda')
IGNORE_INDEX = -1
N_EPOCHS = 100
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 8
TRAIN_INPUT_TRANSFORMS = T.Compose([
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        hue=0.1
    ),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
TRAIN_SHARED_TRANSFORMS = T.Compose([
    T.RandomResizedCrop(
        512,
        scale=(0.5, 2)
    ),
    T.RandomHorizontalFlip(),
    T.Resize((512, 512))
])
# TRAIN_TARGET_TRANSFORMS = T.Compose([
#     T.Resize((512, 512))
# ])
TRAIN_TARGET_TRANSFORMS = None
VAL_INPUT_TRANSFORMS = T.Compose([
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
VAL_SHARED_TRANSFORMS = T.Compose([
    T.Resize((512, 512)),
])


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
    chosen_images = random.sample(image_names, int(len(image_names)*split_ratio))
    for image in image_names:
        if image in  chosen_images:
            shutil.copy(os.path.join(image_dir, image), os.path.join(out_dir, 'labelled', 'image', image))
            shutil.copy(os.path.join(mask_dir, image), os.path.join(out_dir, 'labelled', 'mask', image))
        else:
            shutil.copy(os.path.join(image_dir, image), os.path.join(out_dir, 'unlabelled', 'image', image))


def visualize_pseudo_labels():
    pass


def cutmix(x_U_1, x_U_2=None):
    image_size = x_U_1.shape[-1]
    # init M
    M = np.zeros((image_size, image_size))
    area = random.uniform(0.05, 0.3) * image_size ** 2
    ratio = random.uniform(0.25, 4)
    h = int(np.sqrt(area / ratio))
    w = int(ratio*h)
    start_x = random.randint(0, image_size)
    start_y = random.randint(0, image_size)
    end_x = image_size if start_x+w > image_size else start_x+w
    end_y = image_size if start_y+h > image_size else start_y+h
    M[start_x:end_x, start_y:end_y] += 1
    # cutmix
    # x_U_1: shape bs * c * h * w
    # x_U_2: shape bs * c * h * w
    # x_m: shape bs * c * h * w
    return M


if __name__ == "__main__":
    x_U_1 = np.random.rand(16, 3, 512, 512)
    M = cutmix(x_U_1)
    print(M)
