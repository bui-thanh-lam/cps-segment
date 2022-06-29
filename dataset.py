from torch.utils.data import Dataset
import os
from PIL import Image


class SSLSegmentationDataset(Dataset):
    
    def __init__(
        self,
        image_dir,
        mask_dir=None,
        transform=None,
        target_transform=None,
        return_image_name=False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
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
        with Image.open(os.path.join(self.image_dir, chosen_image)) as img:
            # transformation
            if self.transform is not None:
                img = self.transform(img)
          
        if self.is_unlabelled:
            if self.return_image_name: return img, chosen_image
            else: return img
        else:
            with Image.open(os.path.join(self.mask_dir, chosen_image)) as mask:
                # transformation
                if self.transform is not None:
                    mask = self.target_transform(mask)
            if self.return_image_name: return img, mask.long(), chosen_image
            else: return img, mask.long()


    def __len__(self) -> int:
        return len(self.image_names)
