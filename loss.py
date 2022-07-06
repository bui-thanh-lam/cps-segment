import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE, IGNORE_INDEX


class CombinedCPSLoss(nn.Module):

    def __init__(
            self,
            n_models=3,
            trade_off_factor=1.5,
            pseudo_label_confidence_threshold=0.7,
            use_cutmix=False,
            use_multiple_teachers=False
    ):
        super().__init__()
        self.pseudo_label_confidence_threshold = pseudo_label_confidence_threshold
        self.n_models = n_models
        self.trade_off_factor = trade_off_factor
        self.loss = nn.CrossEntropyLoss(
            reduction='mean',
            ignore_index=IGNORE_INDEX
        )
        self.use_cutmix = use_cutmix
        self.use_multiple_teachers = use_multiple_teachers

    def _multiple_teacher_correction(self, pseudo_labels):
        # shape: bs * h * w * n_models
        _sum = torch.sum(pseudo_labels, dim=-1)
        _pseudo_labels = torch.empty_like(pseudo_labels)
        for i in range(self.n_models):
            _pseudo_labels[:, :, :, :, i] = _sum - pseudo_labels[:, :, :, :, i]
        # shape: bs * h * w * n_models but differnet notations
        return _pseudo_labels

    def _prune_pseudo_label_by_threshold(self, pseudo_labels):
        # pseudo_labels: shape bs * n_classes * h * w * n_models
        _pseudo_labels = torch.max(pseudo_labels, dim=1)[0]
        _pseudo_labels = (torch.exp(_pseudo_labels) >= self.pseudo_label_confidence_threshold).long()
        _pseudo_labels = _pseudo_labels * (torch.argmax(pseudo_labels, dim=1) + 1) - 1
        # _pseudo_labels: shape bs * h * w * n_models
        return _pseudo_labels

    def forward(self, preds_L, preds_U, targets):
        # preds: bs * class * w * h * n_models
        # targets: bs * 1 * w * h
        cps_U_loss = torch.zeros(1).to(DEVICE)
        cps_L_loss = torch.zeros(1).to(DEVICE)
        ce_loss = torch.zeros(1).to(DEVICE)

        if self.use_multiple_teachers:
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
                    # compute cps loss, disable gradient passing
                    with torch.no_grad():
                        # if threshold <= 0.5, don't use threshold clipping
                        if self.pseudo_label_confidence_threshold <= 0.5:
                            Y_U_l = torch.argmax(P_U_l, dim=1)
                            Y_U_r = torch.argmax(P_U_r, dim=1)
                            Y_L_l = torch.argmax(P_L_l, dim=1)
                            Y_L_r = torch.argmax(P_L_r, dim=1)
                        # otherwise, only concern with pseudo labels which have high confidence
                        else:
                            Y_U_l = self._prune_pseudo_label_by_threshold(P_U_l)
                            Y_U_r = self._prune_pseudo_label_by_threshold(P_U_r)
                            Y_L_l = self._prune_pseudo_label_by_threshold(P_L_l)
                            Y_L_r = self._prune_pseudo_label_by_threshold(P_L_r)
                    cps_U_loss += self.loss(P_U_l, Y_U_r) + self.loss(P_U_r, Y_U_l)
                    cps_L_loss += self.loss(P_L_l, Y_L_r) + self.loss(P_L_r, Y_L_l)
            cps_loss = cps_U_loss + cps_L_loss

        # compute supervision loss
        for j in range(self.n_models):
            P_ce_j = preds_L[:, :, :, :, j]
            Y_target = targets.squeeze()
            # ce_loss += self.loss(P_ce_j, Y_target)
            ce_loss += self.loss(P_ce_j, targets)

        # combine two losses
        combined_loss = (ce_loss + float(self.trade_off_factor / (self.n_models - 1)) * cps_loss) / self.n_models
        return combined_loss
