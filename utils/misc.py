
import logging
import itertools
import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter','augment_input']


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import torch.nn as nn
from . import transform_layers as TL

def augment_input(inputs):
    device = inputs.device


    resize_scale = (0.8, 1.0)
    image_size = inputs.shape[-1]

    color_jitter = TL.ColorJitterLayer(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8).to(device)
    color_gray = TL.RandomColorGrayLayer(p=0.2).to(device)
    resize_crop = TL.RandomResizedCropLayer(
        scale=resize_scale, size=image_size).to(device)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    ).to(device)

    return transform(inputs)


