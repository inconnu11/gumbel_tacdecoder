import torch
from torch import nn
import torch.nn.functional as F


class ParrotLoss(nn.Module):
    def __init__(self):
        self.L1Loss = nn.L1Loss()
        # reduction = ???
        # self.MSELoss = F.mse_loss(reduction='mean')

    def forward(self, mel_outputs, feature_predict, ortho_inputs_integral, mask_part, invert_mask):
        loss_ortho_id = self.L1Loss(
            feature_predict.cuda() * invert_mask.cuda() + ortho_inputs_integral.cuda() * self.mask_part.cuda(),
            ortho_inputs_integral.cuda())

        return loss_main_id, loss_ortho_id