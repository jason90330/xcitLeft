# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import numpy as np
import torch.distributed as dist
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def model_eval(output, target, output_dir, epoch):
    target = target.cpu().detach().numpy().copy()
    probablity = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy().copy()
    # probablity = probablity[:,1:]
    score = probablity[:,1:]#np.squeeze(score, 1)
    pred = np.round(score)
    pred = np.array(pred, dtype=int)
    pred = np.squeeze(pred, 1)
    # _, pred = output.topk(1, 1, True, True)
    # pred = pred.t()
    output_dir = output_dir+"txt/"
    if is_main_process() and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  
    if is_main_process():
        with open(output_dir + "score.txt","a+") as f:
            # calculate eer
            
            fpr, tpr, threshold = roc_curve(target,score)          
            fnr = 1-tpr
            diff = np.absolute(fnr - fpr)
            idx = np.nanargmin(diff)
            # print(threshold[idx])
            eer = np.mean((fpr[idx],fnr[idx]))        

            avg = np.add(fpr, fnr)
            idx = np.nanargmin(avg)
            hter = np.mean((fpr[idx],fnr[idx])) 

            fpr_at_10e_m3_idx = np.argmin(np.abs(fpr-10e-3))
            tpr_cor_10e_m3 = tpr[fpr_at_10e_m3_idx+1]

            fpr_at_5e_m3_idx = np.argmin(np.abs(fpr-5e-3))
            print(fpr[-1])
            tpr_cor_5e_m3 = tpr[fpr_at_5e_m3_idx+1]

            fpr_at_10e_m4_idx = np.argmin(np.abs(fpr-10e-4))
            tpr_cor_10e_m4 = tpr[fpr_at_10e_m4_idx+1]

            actual = list(map(lambda el:[el], target))
            pred = list(map(lambda el:[el], pred))
            
            cm = confusion_matrix(actual, pred)
            TP = cm[0][0]
            TN = cm[1][1]
            FP = cm[1][0]
            FN = cm[0][1]
            accuracy = ((TP+TN))/(TP+FN+FP+TN)
            precision = (TP)/(TP+FP)
            recall = (TP)/(TP+FN)
            f_measure = (2*recall*precision)/(recall+precision)
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)		
            error_rate = 1 - accuracy
            apcer = FP/(TN+FP)
            bpcer = FN/(FN+TP)
            acer = (apcer+bpcer)/2
            if is_main_process():
                f.write("="*60)
                f.write('\nModel %03d \n'%(epoch))
                f.write('TP:%d, TN:%d,  FP:%d,  FN:%d\n' %(TP,TN,FP,FN))
                f.write('accuracy:%f\n'%(accuracy))
                f.write('precision:%f\n'%(precision))
                f.write('recall:%f\n'%(recall))
                f.write('f_measure:%f\n'%(f_measure))
                f.write('sensitivity:%f\n'%(sensitivity))
                f.write('specificity:%f\n'%(specificity))
                f.write('error_rate:%f\n'%(error_rate))
                f.write('apcer:%f\n'%(apcer))
                f.write('bpcer:%f\n'%(bpcer))
                f.write('acer:%f\n'%(acer))
                f.write('eer:%f\n'%(eer))
                f.write('hter:%f\n'%(hter))
                f.write('TPR@FPR=10E-3:%f\n'%(tpr_cor_10e_m3))
                f.write('TPR@FPR=5E-3:%f\n'%(tpr_cor_5e_m3))
                f.write('TPR@FPR=10E-4:%f\n\n'%(tpr_cor_10e_m4))

def plot_score(args, modelIdx,fpr, tpr, fnr, roc_auc, cross_data = False, log = False):
    fig = plt.figure()
    lw = 2

    if not cross_data:
        if log:
            plt.xscale("log")
        elif not log:
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylabel('True Living Rate')
    elif cross_data:
        if log:
            plt.xscale("log")
        elif not log:
            plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')#(x0,x1), (y0,y1)
        plt.plot(fpr, fnr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylabel('False Fake Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Living Rate')

    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #fig.savefig('/tmp/roc.png')
    curve_save_path = os.path.join(args.output_dir, "curve")

    if not os.path.exists(curve_save_path):
        os.makedirs(curve_save_path, exist_ok=True)
    if log:
        if cross_data:
            plt.savefig("%s/ROC_cross_log_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
        elif not cross_data:
            plt.savefig("%s/ROC_log_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
    elif not log:
        if cross_data:
            plt.savefig("%s/ROC_cross_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
        elif not cross_data:
            plt.savefig("%s/ROC_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))                