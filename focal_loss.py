import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot_torch(index, classes, cuda):
    '''
    index: torch tensor of size (batch_size * 1), the list of labels, labels are integers and start from 0
    classes: int, # of classes
    cuda: boolean
    '''
    y = index.type(torch.LongTensor)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(y.size()[0], classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.data, 1)
    if cuda: return Variable(y_onehot).cuda()
    else: return Variable(y_onehot)


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target, cuda):
        y = one_hot_torch(target, input.size(-1), cuda)
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = self.alpha * loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum() / input.size()[0]
