from registry import register
from functools import partial
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance, DotProductSimilarity
import torch
registry = {}
register = partial(register, registry=registry)


@register('mse')
class MSELoss():

    def __init__(self, **kwargs):
        self.criterion = torch.nn.MSELoss()

    def __call__(self, target, production):
        return self.criterion(target, production)

def l2norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)


@register('ntx')
class NTXLoss():
    def __init__(self, **kwargs):
        t = 0.07
        if 't' in kwargs:
            t = kwargs['t']
        self.criterion = losses.NTXentLoss(temperature=t)

    def __call__(self, target, production):
        embeddings = torch.cat([target, production], dim=0)

        labels = torch.arange(embeddings.size()[0] // 2)
        labels = torch.cat([labels, labels], dim=0)
        labels.cuda()
        return self.criterion(embeddings, labels)


def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()


def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


@register('align_uniform')
class Align_uniform():
    def __init__(self, alpha=2, t=2, **kwargs):
        self.lam = 1.0
        if 'lam' in kwargs:
            self.lam = kwargs['lam']
        self.alpha = alpha
        self.t = t

    def __call__(self, x, y):
        x, y = l2norm(x), l2norm(y)
        loss = lalign(x, y, self.alpha) + self.lam * (lunif(x, self.t) + lunif(y, self.t)) / 2
        return loss