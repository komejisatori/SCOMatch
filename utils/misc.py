'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter',
           'accuracy_open', 'compute_roc',
           'roc_id_ood', 'ova_ent',
           'test', 'vis']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
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


def accuracy_open(pred, target, topk=(1,), num_classes=5):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)


def roc_id_ood(score_id, score_ood):
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0]+score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all)


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


def vis(args, test_loader, model):

    features = [[] for i in range(7)]
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            _, _, _, feat = model(inputs)

            targets_unk = targets >= 6
            targets[targets_unk] = 6
            known_targets = targets < 6

            for t, f in zip(targets, feat):
                index = t.item()
                features[index].append(f.cpu().numpy())

    return features


def test(args, test_loader, model, num_classes, epoch, val=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_ood = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    top5_ood = AverageMeter()
    end = time.time()
    y_true = []
    y_pred = []
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            _, outputs_pos, outputs_neg, _ = model(inputs)
            outputs = F.softmax(outputs_pos, 1)
            max_ood, _ = torch.max(outputs[:,num_classes:], dim=-1)
            outputs_close = F.softmax(outputs_pos[:,:num_classes], dim=-1)

            pred = outputs[:,:num_classes+1]
            pred[:,-1] = max_ood

            targets_unk = targets >= num_classes
            targets[targets_unk] = num_classes
            known_targets = targets < num_classes
            known_pred = outputs_close[known_targets]
            known_pred_w_ood = pred[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])

                prec1_ood, prec5_ood = accuracy(known_pred_w_ood, known_targets, topk=(1, 5))
                top1_ood.update(prec1_ood.item(), known_pred_w_ood.shape[0])
                top5_ood.update(prec5_ood.item(), known_pred_w_ood.shape[0])


            # pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)

            pseudo_pos = F.softmax(outputs_pos, dim=-1)
            pseudo_id = F.softmax(outputs_pos[:,:num_classes], dim=-1)

            _, targets_p_u = torch.max(pseudo_pos, dim=-1)
            targets_p_u[targets_p_u>num_classes] = num_classes

            max_ood, _ = torch.max(pseudo_pos[:, num_classes:], dim=-1)
            max_id, _ = torch.max(pseudo_id, dim=-1)
            unk_score = max_ood
            unk_close_score = 1 - max_id

            acc_all, unk_acc, size_unk = accuracy_open(targets_p_u,
                                                       targets,
                                                       num_classes=num_classes)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(targets_p_u.cpu().tolist())

            acc.update(acc_all.item(), inputs.shape[0])
            unk.update(unk_acc, size_unk)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx == 0:
                unk_close = unk_close_score
                unk_all = unk_score
                label_all = targets
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                unk_close = torch.cat([unk_close, unk_close_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} "
                                            "acc: {acc:.3f}. "
                                            "unk: {unk:.3f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    acc=acc.avg,
                    unk=unk.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    unk_close = unk_close.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()

    if not val:

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        print('confusion matrix:\n' + np.array_str(cf_mat))

        roc = compute_roc(unk_all, label_all,
                          num_known=num_classes)
        roc_close = compute_roc(unk_close, label_all,
                          num_known=num_classes)

        logger.info("Closed acc: {:.3f}".format(top1.avg))
        logger.info("Overall acc: {:.3f}".format(acc.avg))
        logger.info("Unk acc: {:.3f}".format(unk.avg))
        logger.info("Close roc: {:.3f}".format(roc_close))

        return losses.avg, top1.avg, acc.avg, \
               unk.avg, top1_ood.avg, roc, roc_close
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg

