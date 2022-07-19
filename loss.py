import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.losses import dice_loss

from utils import DEVICE, IGNORE_INDEX


class DiceCELoss(nn.Module):
    """ Mix DiceLoss and Cross Entropy Loss"""
    def __init__(self, dice_weight=1, reduction="mean", ignore_index=IGNORE_INDEX):
        super().__init__()
        self.dice_weight = dice_weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        # ignored element has value of -1
        ce = F.cross_entropy(preds, targets, ignore_index=self.ignore_index, reduction=self.reduction)
        # change all -1 to 0 because of compatiblity
        targets = torch.div((targets + 1), 2, rounding_mode='floor')
        dice = dice_loss(preds, targets)
        return (ce + dice*self.dice_weight) / (1+self.dice_weight)


class CombinedCPSLoss(nn.Module):

    def __init__(
            self,
            n_models=3,
            trade_off_factor=1.5,
            pseudo_label_confidence_threshold=0.7,
            use_cutmix=False,
            use_multiple_teachers=False,
            use_momentum=False
    ):
        super().__init__()
        self.pseudo_label_confidence_threshold = pseudo_label_confidence_threshold
        self.n_models = n_models
        self.trade_off_factor = trade_off_factor
        # self.loss = DiceCELoss()
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='mean')
        self.use_cutmix = use_cutmix
        self.use_multiple_teachers = use_multiple_teachers
        self.use_momentum = use_momentum

    def _multiple_teacher_correction(self, pseudo_labels):
        # shape: bs * n_classes * h * w * n_models
        _sum = torch.sum(pseudo_labels, dim=-1)
        _pseudo_labels = torch.empty_like(pseudo_labels)
        for i in range(self.n_models):
            _pseudo_labels[:, :, :, :, i] = _sum - pseudo_labels[:, :, :, :, i]
        # shape: bs * n_classes * h * w * n_models, but differnet notations
        return _pseudo_labels

    def _prune_pseudo_label_by_threshold(self, pseudo_labels):
        # pseudo_labels: shape bs * n_classes * h * w
        _pseudo_labels = F.softmax(pseudo_labels, dim=1)
        _pseudo_labels = torch.max(_pseudo_labels, dim=1)[0]
        _pseudo_labels = (_pseudo_labels >= self.pseudo_label_confidence_threshold).long()
        _pseudo_labels = _pseudo_labels * (torch.argmax(pseudo_labels, dim=1) + 1) - 1
        # _pseudo_labels: shape bs * h * w
        return _pseudo_labels

    def forward(
        self, 
        targets, 
        preds_L, 
        preds_U=None, 
        preds_U_1=None, 
        preds_U_2=None, 
        preds_m=None, 
        M=None,
        t_preds_L=None, 
        t_preds_U=None, 
        t_preds_U_1=None, 
        t_preds_U_2=None, 
    ):
        # preds: bs * class * w * h * n_models
        # targets: bs * 1 * w * h
        if self.use_cutmix:
            if preds_U_1 is None or preds_U_2 is None or preds_m is None:
                raise ValueError("preds_U_1, preds_U_2, preds_m and M must be provided when use_cutmix=True")
            ce_loss = torch.zeros(1).to(DEVICE)
            cps_loss = torch.zeros(1).to(DEVICE)
            
            if self.use_multiple_teachers:
                if self.use_momentum:
                    Y_1 = self._multiple_teacher_correction(t_preds_U_1)
                    Y_2 = self._multiple_teacher_correction(t_preds_U_2)
                else:
                    Y_1 = self._multiple_teacher_correction(preds_U_1)
                    Y_2 = self._multiple_teacher_correction(preds_U_2)
                for j in range(self.n_models):
                    P_m_j = preds_m[:, :, :, :, j]
                    Y_1_j = Y_1[:, :, :, :, j]
                    Y_2_j = Y_2[:, :, :, :, j]
                    # disable gradient passing
                    with torch.no_grad():
                        ones = torch.ones_like(M)
                        # if threshold <= 0.5, don't use threshold clipping
                        if self.pseudo_label_confidence_threshold <= 0.5:
                            tmp = Y_1_j * (ones - M) + Y_2_j * M
                            Y_j = torch.argmax(tmp, dim=1)
                        # otherwise, only concern with pseudo labels which have high confidence
                        else:
                            tmp = Y_1_j * (ones - M) + Y_2_j * M
                            Y_j = self._prune_pseudo_label_by_threshold(tmp)
                    cps_loss += self.loss(P_m_j, Y_j)

            else:
                for r in range(self.n_models):
                    for l in range(r):
                        if self.use_momentum:
                            P_U_l_1 = t_preds_U_1[:, :, :, :, l]
                            P_U_r_1 = t_preds_U_1[:, :, :, :, r]
                            P_U_l_2 = t_preds_U_2[:, :, :, :, l]
                            P_U_r_2 = t_preds_U_2[:, :, :, :, r]
                        else:
                            P_U_l_1 = preds_U_1[:, :, :, :, l]
                            P_U_r_1 = preds_U_1[:, :, :, :, r]
                            P_U_l_2 = preds_U_2[:, :, :, :, l]
                            P_U_r_2 = preds_U_2[:, :, :, :, r]
                        P_m_l = preds_m[:, :, :, :, l]
                        P_m_r = preds_m[:, :, :, :, r]
                        # compute cps loss, disable gradient passing
                        with torch.no_grad():
                            ones = torch.ones_like(M)
                            # if threshold <= 0.5, don't use threshold clipping
                            if self.pseudo_label_confidence_threshold <= 0.5:
                                tmp = P_U_l_1 * (ones - M) + P_U_l_2 * M
                                Y_l = torch.argmax(tmp, dim=1)
                                tmp = P_U_r_1 * (ones - M) + P_U_r_2 * M
                                Y_r = torch.argmax(P_U_r, dim=1)
                            # otherwise, only concern with pseudo labels which have high confidence
                            else:
                                tmp = P_U_l_1 * (ones - M) + P_U_l_2 * M
                                Y_l = self._prune_pseudo_label_by_threshold(tmp)
                                tmp = P_U_r_1 * (ones - M) + P_U_r_2 * M
                                Y_r = self._prune_pseudo_label_by_threshold(tmp)
                        cps_loss += self.loss(P_m_l, Y_r) + self.loss(P_m_r, Y_l)
        else:
            if preds_L is None or preds_U is None:
                raise ValueError("preds_U and preds_L must be provided when use_cutmix=False")
                
            cps_U_loss = torch.zeros(1).to(DEVICE)
            cps_L_loss = torch.zeros(1).to(DEVICE)
            ce_loss = torch.zeros(1).to(DEVICE)

            if self.use_multiple_teachers:
                if self.use_momentum:
                    Y_U = self._multiple_teacher_correction(t_preds_U)
                    Y_L = self._multiple_teacher_correction(t_preds_L)
                else:
                    Y_U = self._multiple_teacher_correction(preds_U)
                    Y_L = self._multiple_teacher_correction(preds_L)
                for j in range(self.n_models):
                    P_U_j = preds_U[:, :, :, :, j]
                    P_L_j = preds_L[:, :, :, :, j]
                    # disable gradient passing
                    with torch.no_grad():
                        Y_U_j = Y_U[:, :, :, :, j]
                        Y_L_j = Y_L[:, :, :, :, j]
                        if self.pseudo_label_confidence_threshold <= 0.5:
                            Y_U_j = torch.argmax(Y_U_j, dim=1)
                            Y_L_j = torch.argmax(Y_L_j, dim=1)
                        else:
                            Y_U_j = self._prune_pseudo_label_by_threshold(Y_U_j)
                            Y_L_j = self._prune_pseudo_label_by_threshold(Y_L_j)
                    cps_U_loss += self.loss(P_U_j, Y_U_j)
                    cps_L_loss += self.loss(P_L_j, Y_L_j)
                cps_loss = cps_U_loss + cps_L_loss

            else:
                for r in range(self.n_models):
                    for l in range(r):
                        P_U_l = preds_U[:, :, :, :, l]
                        P_U_r = preds_U[:, :, :, :, r]
                        P_L_l = preds_L[:, :, :, :, l]
                        P_L_r = preds_L[:, :, :, :, r]
                        if self.use_momentum:
                            Y_U_l = t_preds_U[:, :, :, :, l]
                            Y_U_r = t_preds_U[:, :, :, :, r]
                            Y_L_l = t_preds_L[:, :, :, :, l]
                            Y_L_r = t_preds_L[:, :, :, :, r]
                        # compute cps loss, disable gradient passing
                        with torch.no_grad():
                            # if threshold <= 0.5, don't use threshold clipping
                            if self.pseudo_label_confidence_threshold <= 0.5:
                                if self.use_momentum:
                                    Y_U_l = torch.argmax(Y_U_l, dim=1)
                                    Y_U_r = torch.argmax(Y_U_r, dim=1)
                                    Y_L_l = torch.argmax(Y_L_l, dim=1)
                                    Y_L_r = torch.argmax(Y_L_r, dim=1)
                                else:
                                    Y_U_l = torch.argmax(P_U_l, dim=1)
                                    Y_U_r = torch.argmax(P_U_r, dim=1)
                                    Y_L_l = torch.argmax(P_L_l, dim=1)
                                    Y_L_r = torch.argmax(P_L_r, dim=1)
                            # otherwise, only concern with pseudo labels which have high confidence
                            else:
                                if self.use_momentum:
                                    Y_U_l = self._prune_pseudo_label_by_threshold(Y_U_l)
                                    Y_U_r = self._prune_pseudo_label_by_threshold(Y_U_r)
                                    Y_L_l = self._prune_pseudo_label_by_threshold(Y_L_l)
                                    Y_L_r = self._prune_pseudo_label_by_threshold(Y_L_r)
                                else:
                                    Y_U_l = self._prune_pseudo_label_by_threshold(P_U_l)
                                    Y_U_r = self._prune_pseudo_label_by_threshold(P_U_r)
                                    Y_L_l = self._prune_pseudo_label_by_threshold(P_L_l)
                                    Y_L_r = self._prune_pseudo_label_by_threshold(P_L_r)
                        cps_U_loss += self.loss(P_U_l, Y_U_r) + self.loss(P_U_r, Y_U_l)
                        cps_L_loss += self.loss(P_L_l, Y_L_r) + self.loss(P_L_r, Y_L_l)
                cps_loss = cps_U_loss + cps_L_loss
        if torch.isnan(cps_loss): cps_loss.zero_()

        # compute supervision loss
        for j in range(self.n_models):
            P_ce_j = preds_L[:, :, :, :, j]
            ce_loss += self.loss(P_ce_j, targets)

        # combine two losses
        combined_loss = (ce_loss + float(self.trade_off_factor / (self.n_models - 1)) * cps_loss) / self.n_models
        return combined_loss
