# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
Modified from: https://github.com/facebookresearch/deit
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from sklearn.metrics import roc_curve, auc
import wandb

def train_log(lr, loss, iter, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"lr": lr, "epoch": epoch, "loss": loss}, step=iter)
    print(f"Loss after " + str(iter).zfill(5) + f" iterations: {loss:.3f}")

def eval_log(acc, loss):
    wandb.log({"accuracy": acc, "loss": loss})

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, surgery=None):
    model.train(set_training_mode)

    if surgery:
        model.module.patch_embed.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    batch_idx=0

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch_idx+=1
        iters = epoch*len(data_loader)+batch_idx
        samples, targets = batch[0], batch[1]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Report metrics every 25th batch
        if ((batch_idx + 1) % 25) == 0:
            train_log(optimizer.param_groups[0]['lr'], loss, iters, epoch)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    outputs = torch.tensor((), device=torch.device('cuda:0'))
    targets = torch.tensor((), device=torch.device('cuda:0'))
    score_list = []
    label_list = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)

        loss = criterion(output, target)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1,))

        output = output.to('cuda:0')
        target = target.to('cuda:0')
        outputs=torch.cat((outputs,output),0)
        targets=torch.cat((targets,target),0)

        label_list = label_list + target.tolist()
        score_list = score_list + output[:,1:].tolist()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    eval_log(metric_logger.acc1.global_avg, metric_logger.loss.global_avg)
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    print("evaluate")

    fpr, tpr, _ = roc_curve(label_list, score_list)
    fnr = 1-tpr
    roc_auc = auc(fpr, tpr) #x,y
    utils.plot_score(args, epoch, fpr, tpr, fnr, roc_auc, cross_data = False, log = False)
    utils.plot_score(args, epoch, fpr, tpr, fnr, roc_auc, cross_data = False, log = True)
    utils.model_eval(outputs, targets, args.output_dir, epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
