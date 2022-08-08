import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from loss import CombinedCPSLoss
from utils import *
from metrics import compute_iou, compute_dice
from models.deeplabv3 import deeplabv3_resnet34, deeplabv3_resnet50, deeplabv3_resnet101
from models.segformer import segformer_b0, segformer_b1, segformer_b2, segformer_b3


class NCPSTrainer:

    def __init__(
        self,
        n_epochs=5,
        device=DEVICE,
        n_models=3,
        model_config=None,
        checkpoint_path=None,
        momentum_factor=0.9,
        trade_off_factor=3,
        n_labelled_examples_per_batch=4,
        pseudo_label_confidence_threshold=0.7,
        learning_rate=1e-4,
        weight_decay=5e-6,
        use_multiple_teachers=False,
        use_cutmix=False,
        use_linear_momentum_scheduler=True,
        use_linear_threshold_scheduler=None,
    ) -> None:
        assert n_models > 1, "number of models should be larger than 1"
        self.n_models = n_models
        self.use_multiple_teachers = use_multiple_teachers
        self.models = []
        self.optims = []
        self.schedulers = []
        self.model_config = model_config
        self.n_labelled_examples_per_batch = n_labelled_examples_per_batch
        self.n_epochs = n_epochs
        self.momentum_factor = momentum_factor
        self.trade_off_factor = trade_off_factor
        self.pseudo_label_confidence_threshold = pseudo_label_confidence_threshold
        self.device = device
        self.use_cutmix = use_cutmix
        if use_linear_momentum_scheduler:
            self.momentum_factor = 0.5
            self.momentum_factor_step = (0.95-0.5) / n_epochs
        else:
            self.momentum_factor_step = None
        if use_linear_threshold_scheduler is not None:
            self.threshold_step = (0.95-0.7) / n_epochs
            self.threshold_scheduler_type = use_linear_threshold_scheduler
            if use_linear_threshold_scheduler == 'increase':
                self.pseudo_label_confidence_threshold = 0.7
            elif use_linear_threshold_scheduler == 'decrease':
                self.pseudo_label_confidence_threshold = 0.9
            else:
                raise ValueError()
        else:
            self.threshold_step = None

        for _ in range(self.n_models):
            if checkpoint_path is not None:
                model = self._register_model(load_pretrained=False)
            else:
                model = self._register_model(load_pretrained=True)
            self.models.append(model.to(self.device))

        self.use_momentum = False
        if momentum_factor > 0:
            self.use_momentum = True
            self.teachers = []
            for i in range(self.n_models):
                self.teachers.append(copy.deepcopy(self.models[i]))

        if checkpoint_path is not None:
            self.load_from_checkpoint(checkpoint_path)

        for i in range(self.n_models):
            self.optims.append(
                torch.optim.AdamW(self.models[i].parameters(), lr=learning_rate, weight_decay=weight_decay))
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optims[i], T_max=self.n_epochs))

    def _register_model(self, load_pretrained=True):
        if self.model_config == "segformer_b0":
            model = segformer_b0(load_pretrained)
        if self.model_config == "segformer_b1":
            model = segformer_b1(load_pretrained)
        if self.model_config == "segformer_b2":
            model = segformer_b2(load_pretrained)
        if self.model_config == "segformer_b3":
            model = segformer_b3(load_pretrained)
        if self.model_config == "deeplabv3_rn34":
            model = deeplabv3_resnet34()
        if self.model_config == "deeplabv3_rn50":
            model = deeplabv3_resnet50()
        if self.model_config == "deeplabv3_rn101":
            model = deeplabv3_resnet101()
        if "segformer" in self.model_config: self.model_type = "transformers"
        if "deeplab" in self.model_config: self.model_type = "torchvision"
        return model

    def _generate_pseudo_labels(self, input):
        for j, model in enumerate(self.models):
            # output logit shape: bs * n_classes * h/4 * w/4
            # Huggingface's Segformer always downscales the output height & width by 4 times
            model.eval()
            if self.model_type == "torchvision":
                p_j = model(input)['out']
            if self.model_type == "transformers":
                p_j = model(input).logits
                p_j = F.interpolate(p_j, scale_factor=4, mode='bilinear')
            p_j = p_j.unsqueeze(-1)
            if j == 0:
                pseudo_labels = p_j
            else:
                pseudo_labels = torch.cat((pseudo_labels, p_j), dim=-1)
            model.train()
        return pseudo_labels

    def _generate_teacher_pseudo_labels(self, input):
        # do not update via backpropagation
        with torch.no_grad():
            for j, model in enumerate(self.teachers):
                # output logit shape: bs * n_classes * h/4 * w/4
                # Huggingface's Segformer always downscales the output height & width by 4 times
                model.eval()

                if self.model_type == "torchvision":
                    p_j = model(input)['out']
                if self.model_type == "transformers":
                    p_j = model(input).logits
                    p_j = F.interpolate(p_j, scale_factor=4, mode='bilinear')
                p_j = p_j.unsqueeze(-1)
                if j == 0:
                    teacher_pseudo_labels = p_j
                else:
                    teacher_pseudo_labels = torch.cat((teacher_pseudo_labels, p_j), dim=-1)
                model.train()

        return teacher_pseudo_labels

    def _cutmix(self, x_U_1, x_U_2):
        # x_U_1: shape bs * c * h * w
        # x_U_2: shape bs * c * h * w
        x_U_1 = x_U_1.numpy()
        x_U_2 = x_U_2.numpy()
        image_size = x_U_1.shape[-1]
        # init M
        M = np.zeros((image_size, image_size))
        area = random.uniform(0.05, 0.3) * image_size ** 2
        ratio = random.uniform(0.25, 4)
        h = int(np.sqrt(area / ratio))
        w = int(ratio * h)
        start_x = random.randint(0, image_size)
        start_y = random.randint(0, image_size)
        end_x = image_size if start_x + w > image_size else start_x + w
        end_y = image_size if start_y + h > image_size else start_y + h
        M[start_x:end_x, start_y:end_y] += 1
        # cutmix
        x_m = x_U_1.copy()
        # x_m: shape bs * c * h * w
        x_m[:, :, start_y:end_y, start_x:end_x] = x_U_2[:, :, start_y:end_y, start_x:end_x]
        x_m = torch.from_numpy(x_m).to(self.device)
        M = torch.from_numpy(M).to(self.device)
        return x_m, M

    def _momentum_update(self):
        # Update teachers' weights
        for i in range(self.n_models):
            for student_weight, teacher_weight in zip(self.models[i].parameters(), self.teachers[i].parameters()):
                teacher_weight.data = teacher_weight.data * (1.0 - self.momentum_factor) + student_weight.data * self.momentum_factor

    def fit(self, labelled_dataset, unlabelled_dataset, val_dataset=None, save_after_one_epoch=False, out_dir=None, logging=True):
        labelled_dataloader = DataLoader(dataset=labelled_dataset, batch_size=self.n_labelled_examples_per_batch)
        unlabelled_dataloader = DataLoader(dataset=unlabelled_dataset, batch_size=self.n_labelled_examples_per_batch)

        criterion = CombinedCPSLoss(
            n_models=self.n_models,
            trade_off_factor=self.trade_off_factor,
            use_cutmix=self.use_cutmix,
            use_multiple_teachers=self.use_multiple_teachers,
            use_momentum=self.use_momentum,
            pseudo_label_confidence_threshold=self.pseudo_label_confidence_threshold,
        ).to(self.device)
        
        for model in self.models:
            model.train()

        if self.use_momentum:
            for teacher in self.teachers:
                teacher.train()
        
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        info = f"Start training, semi-supervised model, model {self.model_config}"
        print(info)
        if logging:
            with open(os.path.join(out_dir, 'log.txt'), 'a+') as log:
                log.write(info+'\n')
        
        for epoch in range(self.n_epochs):
            info = f"======= Epoch {epoch + 1} =======" 
            print(info)
            if logging:
                with open(os.path.join(out_dir, 'log.txt'), 'a+') as log:
                    log.write(info+'\n')
            
            if self.use_cutmix:
                for step, [x_L, Y] in enumerate(labelled_dataloader):
                    x_U_1 = next(iter(unlabelled_dataloader))
                    x_U_2 = next(iter(unlabelled_dataloader))

                    x_m, M = self._cutmix(x_U_1, x_U_2)
                    x_L = x_L.to(self.device)
                    x_U_1 = x_U_1.to(self.device)
                    x_U_2 = x_U_2.to(self.device)
                    Y = Y.to(self.device)

                    # generate pseudo-labels
                    P_m = self._generate_pseudo_labels(x_m)
                    P_U_1 = self._generate_pseudo_labels(x_U_1)
                    P_U_2 = self._generate_pseudo_labels(x_U_2)
                    M = M.expand(P_m.shape[:-1])
                    
                    if self.use_momentum:
                        t_P_U_1 = self._generate_teacher_pseudo_labels(x_U_1)
                        t_P_U_2 = self._generate_teacher_pseudo_labels(x_U_2)

                    # compute loss
                    if self.use_momentum:
                        loss = criterion(
                            preds_L=P_L, 
                            preds_U_1=P_U_1, 
                            preds_U_2=P_U_2, 
                            preds_m=P_m, 
                            targets=Y, 
                            M=M,
                            t_preds_U_1=t_P_U_1,
                            t_preds_U_2=t_P_U_2,
                        )
                    else:
                        loss = criterion(
                            preds_L=P_L, 
                            preds_U_1=P_U_1, 
                            preds_U_2=P_U_2, 
                            preds_m=P_m, 
                            targets=Y, 
                            M=M
                        )
                    if step % 10 == 0:
                        info = f"[INFO] [TRAIN] mode: semi-supervised, epoch: {epoch + 1}, step: {step}, loss: {loss.item():.5f}"
                        print(info)
                        if logging:
                            with open(os.path.join(out_dir, 'log.txt'), 'a+') as log:
                                log.write(info+'\n')

                    # update teacher & student weight
                    loss.backward()
                    if self.use_momentum:
                        self._momentum_update()
                    for i in range(self.n_models):
                        self.optims[i].step()
                        self.optims[i].zero_grad()
            else:
                for step, [x_L, Y] in enumerate(labelled_dataloader):
                    x_U = next(iter(unlabelled_dataloader))

                    x_L = x_L.to(self.device)
                    x_U = x_U.to(self.device)
                    Y = Y.to(self.device)

                    # gen pseudo label
                    P_L = self._generate_pseudo_labels(x_L)
                    P_U = self._generate_pseudo_labels(x_U)
                    if self.use_momentum:
                        t_P_U = self._generate_teacher_pseudo_labels(x_U)

                    # compute loss
                    if self.use_momentum:
                        loss = criterion(
                            preds_L=P_L, 
                            preds_U=P_U, 
                            targets=Y, 
                            t_preds_U=t_P_U,
                        )
                    else:
                        loss = criterion(
                            preds_L=P_L, 
                            preds_U=P_U, 
                            targets=Y, 
                        )

                    if step % 10 == 0:
                        info = f"[INFO] [TRAIN] mode: semi-supervised, epoch: {epoch + 1}, step: {step}, loss: {loss.item():.5f}"
                        print(info)
                        if logging:
                            with open(os.path.join(out_dir, 'log.txt'), 'a+') as log:
                                log.write(info+'\n')

                    # update teacher & student weight
                    loss.backward()
                    if self.use_momentum:
                        self._momentum_update()
                    for i in range(self.n_models):
                        self.optims[i].step()
                        self.optims[i].zero_grad()

            for i in range(self.n_models):
                self.schedulers[i].step()

            if val_dataset is not None:
                info = f"======= Evaluate on ETIS =======\n{self.evaluate(val_dataset)}"
                print(info)
                if logging:
                    with open(os.path.join(out_dir, 'log.txt'), 'a+') as log:
                        log.write(info+'\n')

            if save_after_one_epoch == True and out_dir is not None:
                info = f"Save checkpoint at epoch {epoch+1} to {out_dir}..."
                print(info)
                if logging:
                    with open(os.path.join(out_dir, 'log.txt'), 'a+') as log:
                        log.write(info+'\n')
                self.save(out_dir)

            if self.threshold_step is not None:
                if self.threshold_scheduler_type == 'increase':
                    self.pseudo_label_confidence_threshold += self.threshold_step
                else:
                    self.pseudo_label_confidence_threshold -= self.threshold_step

            if self.momentum_factor_step is not None:
                self.momentum_factor += self.momentum_factor_step

    def evaluate(self, val_dataset, logging_dir=None, dataset_alias=None, mode='soft_voting'):
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)
        dice = []
        iou = []

        for model in self.models:
            model.eval()

        if self.use_momentum:
            for teacher in self.teachers:
                teacher.eval()
        
        if self.use_momentum:
            inference_models = self.teachers
        else:
            inference_models = self.models

        with torch.no_grad():
            for step, [x_L, Y] in enumerate(val_dataloader):
                x_L = x_L.to(self.device)
                Y = Y.to(self.device)
                preds = None
                if mode == "single":
                    if self.model_type == "torchvision":
                        preds = inference_models[0](x_L)['out']
                    if self.model_type == "transformers":
                        preds = inference_models[0](x_L).logits
                if mode == "max_confidence":
                    preds = None
                    for model in inference_models:
                        if self.model_type == "torchvision":
                            _preds = model(x_L)['out']
                        if self.model_type == "transformers":
                            _preds = model(x_L).logits
                        if preds is None:
                            preds = _preds
                        else:
                            preds = torch.maximum(preds, _preds)
                if mode == "soft_voting":
                    preds = None
                    for model in inference_models:
                        if self.model_type == "torchvision":
                            _preds = model(x_L)['out']
                        if self.model_type == "transformers":
                            _preds = model(x_L).logits
                        if preds is None:
                            preds = _preds
                        else:
                            preds += _preds
                if self.model_type == "transformers":
                    preds = F.interpolate(preds, scale_factor=4, mode='bilinear')
    
                if preds.shape != Y.shape:
                    original_size = list(Y.squeeze().shape)
                    preds = F.interpolate(preds, size=original_size, mode='bilinear')
                
                preds = torch.argmax(preds, dim=1).squeeze()
                
                iou.append(compute_iou(preds, Y))
                dice.append(compute_dice(preds, Y))
            mIoU = np.mean(iou)
            mDice = np.mean(dice)
        
        if logging_dir is not None:
            if not os.path.exists(logging_dir):
                os.makedirs(logging_dir)
            info = f'======= Evaluate on {dataset_alias} =======\n"mIoU": {mIoU}, "mDice": {mDice}'
            with open(os.path.join(logging_dir, 'log.txt'), 'a+') as log:
                log.write(info+'\n')
        

        return {
            "mIoU": mIoU,
            "mDice": mDice
        }

    def save(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if self.use_momentum:
            saved_models = self.teachers
        else:
            saved_models = self.models
        for i, model in enumerate(saved_models):
            torch.save(model.state_dict(), os.path.join(out_dir, f'{self.model_config}_head_{i}.pth'))
    
    def load_from_checkpoint(self, checkpoint_path):
        heads = []
        for file in os.listdir(checkpoint_path):
            if file.split('.')[-1] == 'pth':
                heads.append(file)
        for i, head in enumerate(heads):
            _state_dict = torch.load(os.path.join(checkpoint_path, head), self.device)
            self.models[i].load_state_dict(_state_dict)
            if self.use_momentum:
                self.teachers[i].load_state_dict(_state_dict)
            print(f"Load checkpoint from {os.path.join(checkpoint_path, head)} successfully!")

    def predict(self, test_dataset, out_dir, mode='soft_voting'):
        assert mode in ["soft_voting", "max_confidence",
                        "single"], "Mode must be either soft_voting, max_confidence or single"
        assert test_dataset.return_image_name == True, "test_dataset.return_image_name must be True"
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
        
        for model in self.models:
            model.eval()
        if self.use_momentum:
            for teacher in self.teachers:
                teacher.eval()
        
        if self.use_momentum:
            inference_models = self.teachers
        else:
            inference_models = self.models

        with torch.no_grad():
            if test_dataset.is_unlabelled:
                for step, [x, image_name] in enumerate(test_dataloader):
                    if isinstance(image_name, tuple): image_name = image_name[0]
                    x = x.to(self.device)
                    if mode == "single":
                        if self.model_type == "torchvision":
                            preds = self.models[0](x)['out']
                        if self.model_type == "transformers":
                            preds = self.models[0](x).logits
                    if mode == "max_confidence":
                        preds = None
                        for model in inference_models:
                            if self.model_type == "torchvision":
                                _preds = model(x)['out']
                            if self.model_type == "transformers":
                                _preds = model(x).logits
                            if preds is None:
                                preds = _preds
                            else:
                                preds = torch.maximum(preds, _preds)
                    if mode == "soft_voting":
                        preds = None
                        for model in inference_models:
                            if self.model_type == "torchvision":
                                _preds = model(x)['out']
                            if self.model_type == "transformers":
                                _preds = model(x).logits
                            if preds is None:
                                preds = _preds
                            else:
                                preds += _preds
                    if self.model_type == "transformers":
                        preds = F.interpolate(preds, scale_factor=4, mode='bilinear')
                    preds = torch.argmax(preds, dim=1).squeeze()
                    convert_model_output_to_black_and_white_mask(preds, out_dir, image_name)
            else:
                for step, [x, Y, image_name] in enumerate(test_dataloader):
                    if isinstance(image_name, tuple): image_name = image_name[0]
                    x = x.to(self.device)
                    if mode == "single":
                        if self.model_type == "torchvision":
                            preds = self.models[0](x)['out']
                        if self.model_type == "transformers":
                            preds = self.models[0](x).logits
                    if mode == "max_confidence":
                        preds = None
                        for model in self.models:
                            if self.model_type == "torchvision":
                                _preds = model(x)['out']
                            if self.model_type == "transformers":
                                _preds = model(x).logits
                            if preds is None:
                                preds = _preds
                            else:
                                preds = torch.maximum(preds, _preds)
                    if mode == "soft_voting":
                        preds = None
                        for model in self.models:
                            if self.model_type == "torchvision":
                                _preds = model(x)['out']
                            if self.model_type == "transformers":
                                _preds = model(x).logits
                            if preds is None:
                                preds = _preds
                            else:
                                preds += _preds
                    if self.model_type == "transformers":
                        preds = F.interpolate(preds, scale_factor=4, mode='bilinear')
                    preds = torch.argmax(preds, dim=1).squeeze()
                    convert_model_output_to_black_and_white_mask(preds, out_dir, image_name)
