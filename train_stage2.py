import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.filters import threshold_otsu
from dataset.dataset_mismatch import DATASET_GETTERS_MISMATCH
from utils import AverageMeter, accuracy, augment_input
from models.ema import ModelEMA

import tempfile
import os

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def main():
    parser = argparse.ArgumentParser(description='Stage2 Training with Mismatch Ratio')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'tinyimagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=400,
                        help='number of labeled data per class')
    parser.add_argument('--num-val', type=int, default=20,
                        help='number of validation data per class')
    parser.add_argument('--mismatch-ratio', type=float, default=0.3,
                        help='mismatch ratio (OOD samples ratio in unlabeled data)')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--mu', default=5, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--opt_level", type=str, default="O1")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--filter-every-epoch', type=int, default=20, 
                        help='every K epoch to filter in distribution unlabeled data')

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    if args.dataset == 'cifar10':
        args.num_classes = 6
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
    elif args.dataset == 'cifar100':
        args.num_classes = 50
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
    elif args.dataset == 'mnist':
        args.num_classes = 6
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    args.epochs = math.ceil(args.total_steps / args.eval_step)

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))

        rotnet_head = torch.nn.Linear(64*args.model_width, 4)
        feat_dim = 64 * args.model_width
        cons_proj = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feat_dim, feat_dim)
        )
        rot_proj = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feat_dim, feat_dim)
        )
        return model, rotnet_head, rot_proj, cons_proj

    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = DATASET_GETTERS_MISMATCH[args.dataset](args)

    udst_rotnet = deepcopy(unlabeled_dataset)
    udst_rotnet.transform = labeled_dataset.transform

    udst_eval = deepcopy(unlabeled_dataset)
    udst_eval.transform = test_dataset.transform
    udst_eval_loader = DataLoader(
        udst_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size*args.mu,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    udst_rotnet_loader = DataLoader(
        udst_rotnet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    
    model, rotnet_head, rot_proj, cons_proj = create_model(args)
    model, rotnet_head, rot_proj, cons_proj = model.to(args.device), rotnet_head.to(args.device), rot_proj.to(args.device), cons_proj.to(args.device)

    # Initialize EMA
    ema_model = None
    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)

    # Load Stage1 checkpoint if provided
    if args.resume and os.path.isfile(args.resume):
        logger.info("==> Loading Stage1 checkpoint..")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        rotnet_head.load_state_dict(checkpoint['rotnet_state_dict'])
        rot_proj.load_state_dict(checkpoint['rot_proj_state_dict'])
        cons_proj.load_state_dict(checkpoint['cons_proj_state_dict'])
        if args.use_ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        logger.info("==> Stage1 checkpoint loaded successfully")

    train_stage2(args, labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, 
                 udst_rotnet_loader, udst_eval_loader, unlabeled_dataset,
                 model, rotnet_head, rot_proj, cons_proj, ema_model)


def train_stage2(args, labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, 
                 udst_rotnet_loader, udst_eval_loader, unlabeled_dataset,
                 model, rotnet_head, rot_proj, cons_proj, ema_model):


    global best_acc, best_acc_val
    val_accs = []
    test_accs = []
    end = time.time()

    grouped_parameters = [
        {'params': model.parameters()},
        {'params': rotnet_head.parameters()},
        {'params': rot_proj.parameters()},
        {'params': cons_proj.parameters()}
    ]

    optimizer = optim.SGD(grouped_parameters, lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    rotnet_iter = iter(udst_rotnet_loader)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_r = AverageMeter()

        # clean unlabeled data periodically for consistency constraint loss
        if epoch % args.filter_every_epoch == 0:
            in_dist_idxs = filter_ood(args, udst_eval_loader, model, cons_proj)
            in_dist_unlabeled_dataset = Subset(unlabeled_dataset, in_dist_idxs)
            unlabeled_trainloader = DataLoader(
                in_dist_unlabeled_dataset,
                batch_size=args.batch_size*args.mu,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True)

        model.train()
        rotnet_head.train()
        rot_proj.train()
        cons_proj.train()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x, index_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, index_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), gt_u, index_u = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), gt_u, index_u = next(unlabeled_iter)
            
            try:
                inputs_r, gt_u_r, index_u_r = next(rotnet_iter)
            except:
                rotnet_iter = iter(udst_rotnet_loader)
                inputs_r, gt_u_r, index_u_r = next(rotnet_iter)

            # rotate unlabeled data with 0, 90, 180, 270 degrees
            inputs_r = torch.cat(
                [torch.rot90(inputs_r, i, [2, 3]) for i in range(4)], dim=0)
            targets_r = torch.cat(
                [torch.empty(index_u_r.size(0)).fill_(i).long() for i in range(4)], dim=0).to(args.device)

            data_time.update(time.time() - end)
            
            batch_size = inputs_x.shape[0]
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            targets_x = targets_x.to(args.device)

            logits, feats = model(inputs, output_feats=True)

            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            feats_x = feats[:batch_size]

            # Cross Entropy Loss for Labeled Data
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # consistency constraint loss for unlabeled data
            T = 0.4
            p_cutoff = 0.8
            logits_tgt = logits_u_w / T
            probs_u_w = torch.softmax(logits_u_w, dim=1)
            loss_mask = probs_u_w.max(-1)[0].ge(p_cutoff)

            if loss_mask.sum() == 0:
                Lu = torch.zeros(1, dtype=torch.float).to(args.device)
            else:
                Lu = F.kl_div(
                    torch.log_softmax(logits_u_s[loss_mask], -1), 
                    torch.softmax(logits_tgt[loss_mask].detach().data, -1),
                    reduction='batchmean')

            # Cross Entropy Loss for Rotation Recognition
            inputs_r = inputs_r.to(args.device)
            logits_r, feats_r = model(inputs_r, output_feats=True)
            Lr = F.cross_entropy(rotnet_head(rot_proj(feats_r)), targets_r, reduction='mean')

            loss = Lx + Lr + Lu

            optimizer.zero_grad()
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_r.update(Lr.item())
            losses_u.update(Lu.item())

            optimizer.step()
            scheduler.step()
            
            # EMA update
            if args.use_ema and ema_model is not None:
                ema_model.update(model)

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. "
                                      "Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_r: {loss_r:.4f}. Loss_u: {loss_u:.4f}. "
                                      "Loss_u: {loss_u:.4f}".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_r=losses_r.avg,
                    loss_u=losses_u.avg,
                ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        # Use EMA model for testing if available
        test_model = ema_model.ema if (args.use_ema and ema_model is not None) else model

        val_loss, val_acc = test(args, val_loader, test_model, epoch)
        test_loss, test_acc = test(args, test_loader, test_model, epoch)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_r', losses_r.avg, epoch)
        args.writer.add_scalar('train/4.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
        args.writer.add_scalar('val/1.val_acc', val_acc, epoch)
        args.writer.add_scalar('val/2.val_loss', val_loss, epoch)

        best_acc_val = max(val_acc, best_acc_val)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        model_to_save = model.module if hasattr(model, "module") else model
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_model.ema.state_dict() if (args.use_ema and ema_model is not None) else None,
            'rotnet_state_dict': rotnet_head.state_dict(),
            'rot_proj_state_dict': rot_proj.state_dict(),
            'cons_proj_state_dict': cons_proj.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        val_accs.append(val_acc)
        logger.info('Best top-1 acc(test): {:.2f} | acc(val): {:.2f}'.format(best_acc, best_acc_val))
        logger.info('Mean top-1 acc(test): {:.2f} | acc(val): {:.2f}\n'.format(
            np.mean(test_accs[-20:]), np.mean(val_accs[-20:])))


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):

            if len(batch_data) == 3:
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


def filter_ood(args, loader, model, cons_proj):
    """
    use consistency score to filter OOD samples
    """
    # switch to evaluate mode
    model.eval()
    cons_proj.eval()
    matching_scores = []
    targets = []
    idxs = []
    in_dist_idxs = []
    ood_cnt = 0

    with torch.no_grad():
        for batch_idx, (input, target, indexs) in enumerate(loader):
            input = input.to(args.device)
            input_aug = augment_input(input)

            _, feats = model(input, output_feats=True)
            _, feats_aug = model(input_aug, output_feats=True)

            proj = cons_proj(feats)
            proj_aug = cons_proj(feats_aug)

            # compute MSE as consistency score
            score = -F.mse_loss(proj, proj_aug, reduction='none').mean(dim=1)
            matching_scores.extend(score.cpu().tolist())
            idxs.extend(indexs.tolist())
            targets.extend(target.tolist())

    # use otsu threshold to adaptively compute threshold
    matching_scores = np.array(matching_scores)
    thresh = threshold_otsu(matching_scores)
    for i, s in enumerate(matching_scores):
        if s > thresh:
            in_dist_idxs.append(idxs[i])
            if targets[i] == -1:
                ood_cnt += 1
    logger.info('OOD Filtering threshold: %.3f' % thresh)
    logger.info('false positive: %d/%d' % (ood_cnt, len(in_dist_idxs)))

    model.train()
    cons_proj.train()
    return in_dist_idxs


if __name__ == '__main__':
    main() 