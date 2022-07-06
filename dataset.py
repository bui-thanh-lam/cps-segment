from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as T
import os
import numpy as np
import torch
from transformers import SegformerFeatureExtractor


class SSLSegmentationDataset(Dataset):

    def __init__(
            self,
            image_dir,
            mask_dir=None,
            input_transform=None,
            target_transform=None,
            shared_transform=None,
            return_image_name=False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.shared_transform = shared_transform
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.feature_extractor.reduce_labels = False
        self.return_image_name = return_image_name

        if mask_dir is None:
            self.is_unlabelled = True
        else:
            mask_names = os.listdir(mask_dir)
            self.mask_names = []
            self.is_unlabelled = False

        # Only consider which image names exist in both image_dir and mask_dir
        self.image_names = []
        for file in os.listdir(image_dir):
            [name, ext] = file.split(".")
            if ext in ["png", "jpg"]:
                if self.is_unlabelled:
                    self.image_names.append(file)
                elif file in mask_names:
                    self.image_names.append(file)
                    self.mask_names.append(file)

    def __getitem__(self, index: int):
        chosen_image = self.image_names[index]

        if self.is_unlabelled:
            img = Image.open(os.path.join(self.image_dir, chosen_image))
            # transformation
            # img = np.asarray(img)
            # img = torch.FloatTensor(img).permute(2, 0, 1)
            img = self.feature_extractor(img, return_tensors='pt')["pixel_values"].squeeze()
            if self.input_transform is not None:
                img = self.input_transform(img)
            if self.shared_transform is not None:
                raise ValueError("shared_transform should be None for unlabelled dataset")
            if self.return_image_name:
                return img, chosen_image
            else:
                return img
        else:
            img = Image.open(os.path.join(self.image_dir, chosen_image))
            mask = Image.open(os.path.join(self.mask_dir, chosen_image))
            # transformation
            # mask = np.asarray(mask)
            # mask = torch.LongTensor(mask)
            features = self.feature_extractor(img, mask, return_tensors='pt')
            img = features["pixel_values"].squeeze()
            mask = features["labels"]
            if self.shared_transform is not None:
                example = torch.cat((img, mask), dim=0)
                example = self.shared_transform(example)
                img = example[:3, :, :]
                mask = example[-1, :, :].unsqueeze(0)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            mask = mask.squeeze()
            mask = mask.type(torch.int64)
            if self.return_image_name:
                return img, mask, chosen_image
            else:
                return img, mask

    def __len__(self) -> int:
        return len(self.image_names)
