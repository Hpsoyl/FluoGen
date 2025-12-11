import torch

from torch import nn

#pinball loss class
class PinballLoss():

  def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

  def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


def quantile_regression_loss_fn(pred, target, args):
    q_lo_loss = PinballLoss(args.q_lo)
    q_hi_loss = PinballLoss(args.q_hi)

    loss = args.q_lo_weight * q_lo_loss(pred[:,0,:,:].squeeze(), target.squeeze()) + \
            args.q_hi_weight * q_hi_loss(pred[:,2,:,:].squeeze(), target.squeeze())
 
    return loss



