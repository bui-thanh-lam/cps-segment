import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from loss import CombinedCPSLoss, DiceLoss
from utils import *
from evaluate import compute_iou, compute_dice


class NCPSTrainer:

    def __init__(
        self,
        n_epochs,
        device = DEVICE,
        n_models = 3,
        model_config: SegformerConfig = None,
        model: SegformerForSemanticSegmentation = None,
        momentum_factor = 0,
        trade_off_factor = 1.5,
        n_labelled_examples_per_batch = 4,
        learning_rate = 5e-5,
        weight_decay=5e-6,
        use_multiple_teachers = False,
        use_cutmix = False
    ) -> None:
        assert n_models > 1, "number of models should be larger than 1"
        self.n_models = n_models
        self.use_multiple_teachers = use_multiple_teachers
        self.models = []
        self.optims = []
        self.model_config = model_config
        self.n_labelled_examples_per_batch = n_labelled_examples_per_batch
        self.n_epochs = n_epochs
        self.use_cutmix = use_cutmix
        self.momentum_factor = momentum_factor
        self.trade_off_factor = trade_off_factor
        self.device = device

        if model is not None:
            for _ in range(self.n_models):
                self.models.append(model.deepcopy().to(self.device))
        elif model_config is not None:
            for _ in range(self.n_models):
                self.models.append(SegformerForSemanticSegmentation(config=model_config).to(self.device))
        else:
            raise ValueError("Either model_config or model must be not None")

        for i in range(self.n_models):
            self.optims.append(torch.optim.AdamW(self.models[i].parameters(), lr=learning_rate, weight_decay=weight_decay))


    def _generate_pseudo_labels(self, input):
        for j in range(self.n_models):
            # output logit shape: bs * n_classes * h/4 * w/4
            # Huggingface's Segformer always downscales the output height & width by 4 times
            p_j = self.models[j](input).logits
            p_j = F.interpolate(p_j, scale_factor=4, mode='bicubic').unsqueeze(-1)
            if j == 0: pseudo_labels = p_j
            else: pseudo_labels = torch.cat((pseudo_labels, p_j), dim=-1)
        # shape: bs * h * w * n_models
        return pseudo_labels


    def _cutmix(self, x_U_1, x_U_2):
        # init M
        # cutmix
        pass

    
    def _multiple_teacher_correction(self, pseudo_labels):
        pass


    def _momentum_update(self):
        pass


    def fit(self, labelled_dataset, unlabelled_dataset):
        labelled_dataloader = DataLoader(dataset=labelled_dataset, batch_size=self.n_labelled_examples_per_batch)
        unlabelled_dataloader = DataLoader(dataset=unlabelled_dataset, batch_size=self.n_labelled_examples_per_batch)
        for epoch in range(self.n_epochs):
            print(f"======= Epoch {epoch+1} =======")
            criterion = CombinedCPSLoss(
                n_models=self.n_models,
                trade_off_factor=self.trade_off_factor,
            ).to(self.device)
            for (step, [x_L, Y]), (_, x_U) in zip(enumerate(labelled_dataloader), enumerate(unlabelled_dataloader)):
                x_L = x_L.to(self.device)
                x_U = x_U.to(self.device)
                Y = Y.to(self.device)
                # gen pseudo label
                P_L = self._generate_pseudo_labels(x_L)
                P_U = self._generate_pseudo_labels(x_U)
                # compute loss
                loss = criterion(preds_sup=P_L, preds_unsup=P_U, targets=Y)
                if step % 50 == 0:
                    print(f"{loss.item():.5f}")
                # update teacher & student weight
                loss.backward()
                for i in range(self.n_models):
                    self.optims[i].step()
                    self.optims[i].zero_grad()
            print(f"======= Evaluate on train set =======")
            print(self.evaluate(labelled_dataset))

    
    def evaluate(self, val_dataset):
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)
        dice = []
        iou = []
        with torch.no_grad():
            for step, [x_L, Y] in enumerate(val_dataloader):
                x_L = x_L.to(self.device)
                Y = Y.to(self.device).squeeze()
                preds = self.models[0](x_L).logits
                preds = F.interpolate(preds, scale_factor=4, mode='bicubic')
                preds = torch.argmax(preds, dim=1).squeeze()
                iou.append(compute_iou(preds, Y))
                dice.append(compute_dice(preds, Y))
            mIoU = np.mean(iou)
            mDice = np.mean(dice)
        return {
            "mIoU": mIoU,
            "mDice": mDice
        }

    
    def save(self):
        pass


    def predict(self):
        pass


class SupervisedTrainer:

    def __init__(
        self,
        n_epochs,
        device = DEVICE,
        model_config: SegformerConfig = None,
        model: SegformerForSemanticSegmentation = None,
        learning_rate = 5e-5,
        weight_decay=5e-6,
        batch_size=16
    ) -> None:
        self.n_epochs = n_epochs
        self.device = device
        self.batch_size = batch_size
        if model is not None:
            self.model_config = None
            self.model = model.deepcopy().to(self.device)
        elif model_config is not None:
            self.model_config = model_config
            self.model = SegformerForSemanticSegmentation(config=model_config).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    def fit(self, labelled_dataset, test_dataset=None):
        labelled_dataloader = DataLoader(dataset=labelled_dataset, batch_size=self.batch_size)
        for epoch in range(self.n_epochs):
            print(f"======= Epoch {epoch+1} =======")
            # criterion = DiceLoss().to(self.device)
            criterion = nn.CrossEntropyLoss().to(self.device)
            for step, [x_L, Y] in enumerate(labelled_dataloader):
                x_L = x_L.to(self.device)
                Y = Y.squeeze().to(self.device)
                preds = self.model(x_L).logits
                preds = F.interpolate(preds, scale_factor=4, mode='bicubic')
                loss = criterion(preds, Y)
                if step % 50 == 0:
                    print(f"{loss.item():.5f}")
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            print(f"======= Evaluate on train set =======")
            print(self.evaluate(labelled_dataset))
            self.predict(test_dataset, "datasets/TestDataset/CVC-300/output/small/")



    def evaluate(self, val_dataset):
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)
        dice = []
        iou = []
        with torch.no_grad():
            for step, [x_L, Y] in enumerate(val_dataloader):
                x_L = x_L.to(self.device)
                Y = Y.to(self.device).squeeze()
                preds = self.model(x_L).logits
                preds = F.interpolate(preds, scale_factor=4, mode='bicubic')
                preds = torch.argmax(preds, dim=1).squeeze()
                iou.append(compute_iou(preds, Y))
                dice.append(compute_dice(preds, Y))
            mIoU = np.mean(iou)
            mDice = np.mean(dice)
        return {
            "mIoU": mIoU,
            "mDice": mDice
        }

    
    def save(self):
        pass


    def predict(self, test_dataset, out_dir):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
        with torch.no_grad():
            for step, [x_L, Y, image_name] in enumerate(test_dataloader):
                image_name = image_name[0]
                x_L = x_L.to(self.device)
                Y = Y.to(self.device).squeeze()
                preds = self.model(x_L).logits
                preds = F.interpolate(preds, scale_factor=4, mode='bicubic')
                preds = torch.argmax(preds, dim=1).squeeze()
                convert_model_output_to_black_and_white_mask(preds, out_dir, image_name)