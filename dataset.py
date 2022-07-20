from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import os
import numpy as np
import torch
from transformers import SegformerFeatureExtractor

from utils import IMAGE_SIZE


class SSLSegmentationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir=None,
        feature_extractor_config=None,
        input_transform=None,
        target_transform=None,
        shared_transform=None,
        return_image_name=False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor_config = feature_extractor_config
        self._register_feature_extractor()
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.shared_transform = shared_transform
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
                    
    def _register_feature_extractor(self):
        # huggingface's feature extractor only does resizing & normalization
        if "segformer" not in self.feature_extractor_config: return
        if self.feature_extractor_config == "segformer_b0":
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
        if self.feature_extractor_config == "segformer_b1":
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b1")
        if self.feature_extractor_config == "segformer_b2":
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b2")
        if self.feature_extractor_config == "segformer_b3":
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b3")
        self.feature_extractor.reduce_labels = False

    def __getitem__(self, index: int):
        chosen_image = self.image_names[index]

        if self.is_unlabelled:
            img = Image.open(os.path.join(self.image_dir, chosen_image))
            # transformation
            if self.shared_transform is not None:
                raise ValueError("shared_transform should be None for unlabelled dataset")
            if "segformer" in self.feature_extractor_config:
                img = self.feature_extractor(img, return_tensors='pt')["pixel_values"].squeeze()
            else:
                img = np.asarray(img)
                img = torch.FloatTensor(img).permute(2, 0, 1)
                img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.input_transform is not None:
                img = self.input_transform(img)
            if self.return_image_name:
                return img, chosen_image
            else:
                return img
        else:
            img = Image.open(os.path.join(self.image_dir, chosen_image))
            mask = Image.open(os.path.join(self.mask_dir, chosen_image))
            # transformation
            if "segformer" in self.feature_extractor_config:
                features = self.feature_extractor(img, mask, return_tensors='pt')
                img = features["pixel_values"].squeeze()
                mask = features["labels"]
            else:
                img = np.asarray(img)
                img = torch.FloatTensor(img).permute(2, 0, 1)
                mask = np.asarray(mask)
                mask = torch.FloatTensor(mask).unsqueeze(0)
                # only resize & normalize labelled data
                img = F.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                mask = F.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
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
