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
    ):
        super().__init__()
        self.pseudo_label_confidence_threshold = pseudo_label_confidence_threshold
        self.n_models = n_models
        self.trade_off_factor = trade_off_factor
        self.loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([0.1, 1.0]),
            reduction='mean',
            ignore_index=IGNORE_INDEX
        )


    def _prune_pseudo_label_by_threshold(self, pseudo_labels):
        # pseudo_labels: shape bs * n_classes * h * w * n_models
        _pseudo_labels = torch.max(pseudo_labels, dim=1)[0]
        _pseudo_labels = (torch.exp(_pseudo_labels) >= self.pseudo_label_confidence_threshold).long()
        _pseudo_labels = _pseudo_labels * (torch.argmax(pseudo_labels, dim=1) + 1) - 1
        # _pseudo_labels: shape bs * h * w * n_models
        return _pseudo_labels


    def forward(self, preds_sup, preds_unsup, targets):
        # preds: bs * class * w * h * n_models
        # targets: bs * 1 * w * h
        unsup_cps_loss = torch.zeros(1).to(DEVICE)
        sup_cps_loss = torch.zeros(1).to(DEVICE)
        ce_loss = torch.zeros(1).to(DEVICE)
        
        # compute cps loss, disable gradient passing
        for r in range(self.n_models):
            for l in range(r):
                P_unsup_l = preds_unsup[:, :, :, :, l].squeeze()
                P_unsup_r = preds_unsup[:, :, :, :, r].squeeze()
                P_sup_l = preds_sup[:, :, :, :, l].squeeze()
                P_sup_r = preds_sup[:, :, :, :, r].squeeze()
                with torch.no_grad():
                    # if threshold <= 0.5, don't use thresold clipping
                    if self.pseudo_label_confidence_threshold <= 0.5:
                        Y_unsup_l = torch.argmax(P_unsup_l, dim=1)
                        Y_unsup_r = torch.argmax(P_unsup_r, dim=1)
                        Y_sup_l = torch.argmax(P_sup_l, dim=1)
                        Y_sup_r = torch.argmax(P_sup_r, dim=1)
                    # otherwise, only concern with pseudo labels which have high confidence
                    else:
                        Y_unsup_l = self._prune_pseudo_label_by_threshold(P_unsup_l)
                        Y_unsup_r = self._prune_pseudo_label_by_threshold(P_unsup_r)
                        Y_sup_l = self._prune_pseudo_label_by_threshold(P_sup_l)
                        Y_sup_r = self._prune_pseudo_label_by_threshold(P_sup_r)
                unsup_cps_loss += self.loss(P_unsup_l, Y_unsup_r) + self.loss(P_unsup_r, Y_unsup_l)
                sup_cps_loss += self.loss(P_sup_l, Y_sup_r) + self.loss(P_sup_r, Y_sup_l)
        cps_loss = unsup_cps_loss + sup_cps_loss

        # compute supervision loss
        for j in range(self.n_models):
            P_ce_j = preds_sup[:, :, :, :, j].squeeze()
            Y_target = targets.squeeze()
            ce_loss += self.loss(P_ce_j, Y_target)
        
        # combine two losses
        combined_loss = (ce_loss + float(self.trade_off_factor / (self.n_models-1)) * cps_loss) / self.n_models
        return combined_loss


class DiceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    
    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        # inputs = torch.max(inputs, dim=1)[0].squeeze()
        inputs = inputs.view(-1)
        targets = F.one_hot(targets, 2).permute(0, 3, 2, 1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice