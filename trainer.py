import logging
import time
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from utils import AverageMeter, save_checkpoint, test, wandb_logger

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader, model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp

    global best_acc
    global best_acc_val

    num_classes = args.num_classes
    test_accs = []

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0

    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "thr: {thr:.2f}. "
    output_args = vars(args)
    wandb_logger.init({"method": "SCOMatch"})

    model.train()
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    labeled_dataset = copy.deepcopy(labeled_trainloader.dataset)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset, replacement=True),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                       sampler=train_sampler(unlabeled_dataset),
                                       batch_size=int(args.batch_size * args.mu),
                                       num_workers=args.num_workers,
                                       drop_last=True)
    unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                           sampler=train_sampler(unlabeled_dataset_all),
                                           batch_size=int(args.batch_size * args.mu),
                                           num_workers=args.num_workers,
                                           drop_last=True)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    unlabeled_all_iter = iter(unlabeled_trainloader_all)

    selected_ood_maxlength = max(8 * args.num_classes, 256)
    selected_ood_update_length = args.Km
    selected_ood_count = 0
    selected_ood_scores = deque(maxlen=selected_ood_maxlength)
    selected_ood_labels = deque(maxlen=selected_ood_maxlength)
    selected_ood_images = deque(maxlen=selected_ood_maxlength)

    all_sample_scores = [[]for i in range(num_classes+1)]
    ood_threshold = args.ood_threshold
    threshold_update_freq = (len(unlabeled_dataset_all)) // int(args.batch_size * args.mu * 2)

    for epoch in range(args.start_epoch, args.epochs):

        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        for batch_idx in range(args.eval_step):

            if batch_idx % threshold_update_freq == 0 and batch_idx > 0 and epoch >= args.start_fix:
                max_len = sum([len(all_sample_scores[i]) for i in range(num_classes)])
                ood_len = len(all_sample_scores[-1])
                if max_len > 0:
                    ratio = ood_len / (max_len)
                    ood_threshold = args.threshold * (ratio)
                    ood_threshold = min(0.95, max(0.75, ood_threshold))
                else:
                    ood_threshold = args.ood_threshold
                all_sample_scores = [[] for i in range(num_classes + 1)]

            ## Data loading
            try:
                (inputs_x, inputs_x_s, inputs_x_w), targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_s, inputs_x_w), targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s, _), targets_uc_eval = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), targets_uc_eval = unlabeled_iter.next()

            try:
                (inputs_all_w, inputs_all_s, _), targets_u_eval = unlabeled_all_iter.next()
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, _), targets_u_eval = unlabeled_all_iter.next()

            data_time.update(time.time() - end)
            b_size = inputs_x.shape[0]
            acc = 0

            if selected_ood_count < args.batch_size:
                inputs = torch.cat([
                    inputs_x,
                    inputs_all_w, inputs_all_s,
                    inputs_u_w, inputs_u_s
                ], 0).to(args.device)
                _, logits_p, _, _ = model(inputs)
                logits_id_lb = logits_p[:b_size]
                logits_open_w, logits_open_s, logits_close_w, logits_close_s = logits_p[b_size:].chunk(4)
                L_sup_open = torch.zeros(1).to(args.device).mean()
            else:
                indices = torch.randperm(len(selected_ood_images))[:args.batch_size]
                ood_samples = torch.stack(list(selected_ood_images))[indices]
                ood_scores = torch.tensor(list(selected_ood_scores))[indices].to(args.device)
                ood_label = (torch.ones(args.batch_size) * num_classes).to(args.device).long()

                inputs = torch.cat([
                    inputs_x, ood_samples,
                    inputs_all_w, inputs_all_s,
                    inputs_u_w, inputs_u_s
                ], 0).to(args.device)

                _, logits_p, _, _ = model(inputs)
                logits_id_lb = logits_p[:b_size]
                logits_ood_lb = logits_p[b_size:b_size+args.batch_size]
                logits_open_w, logits_open_s, logits_close_w, logits_close_s = logits_p[b_size+args.batch_size:].chunk(4)

                ood_mask = ood_scores < args.threshold
                L_sup_open = (
                        F.cross_entropy(logits_ood_lb, ood_label, reduction='none')
                        * ood_mask
                ).mean()

            targets_x = targets_x.to(args.device)
            targets_uc_eval = targets_uc_eval.to(args.device)
            L_sup_close = F.cross_entropy(logits_id_lb, targets_x, reduction='mean')

            pseudo_label_open = torch.softmax(logits_open_w.detach() / args.T, dim=-1)
            max_probs, targets_u_all = torch.max(pseudo_label_open, dim=-1)
            for prob, target in zip(max_probs, targets_u_all):
                if prob > args.threshold:
                    all_sample_scores[target.item()].append(prob.item())

            max_probs_open, _ = torch.max(pseudo_label_open[:, :num_classes], dim=-1)
            _, indices = torch.sort(max_probs_open)
            indices = indices[:selected_ood_update_length]
            if selected_ood_count < selected_ood_maxlength:
                selected_ood_count += selected_ood_update_length
            for prob, img, ulab in zip(max_probs_open[indices], inputs_all_w[indices], targets_u_eval[indices]):
                selected_ood_scores.append(prob.item())
                selected_ood_images.append(img)
                selected_ood_labels.append(ulab.item())

            max_probs_open, targets_u_all_open = torch.max(pseudo_label_open, dim=-1)
            mask_pos = max_probs_open.ge(args.threshold) & (targets_u_all_open < num_classes)
            mask_pos = mask_pos | ((max_probs_open.ge(ood_threshold)) & (targets_u_all_open == num_classes))
            if args.dataset == 'cifar10':
                L_unsup_open = (
                        F.cross_entropy(
                            torch.cat([logits_open_s], dim=0), targets_u_all_open,
                            reduction='none') * mask_pos
                ).mean()
            else:
                L_unsup_open = (
                        F.cross_entropy(
                            torch.cat([logits_open_w, logits_open_s], dim=0), targets_u_all_open.repeat(2),
                            reduction='none') * mask_pos.repeat(2)
                ).mean()


            logits_p_u_close_w = logits_close_w[:, :num_classes]
            logits_p_u_close_s = logits_close_s[:, :num_classes]

            pseudo_close = torch.softmax(logits_p_u_close_w.detach() / args.T, dim=-1)
            pseudo_open = torch.softmax(logits_close_w.detach() / args.T, dim=-1)

            max_probs_close, targets_close = torch.max(pseudo_close, dim=-1)
            max_probs_open, targets_open = torch.max(pseudo_open, dim=-1)

            mask = max_probs.ge(args.threshold).float()
            id_mask = (targets_open < num_classes)
            L_unsup_close = (F.cross_entropy(logits_p_u_close_s,
                                     targets_close,
                                     reduction='none') * (mask * id_mask)).mean()

            if epoch < args.start_fix:

                L_unsup_open = torch.zeros(1).to(args.device).mean()
                L_sup_open = torch.zeros(1).to(args.device).mean()

            loss = L_sup_close + L_sup_open + L_unsup_close + L_unsup_open

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())

            output_args["batch"] = batch_idx
            output_args["thr"] = ood_threshold
            output_args["loss"] = losses.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]

            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            val_test_acc_close = test(args, val_loader, test_model, num_classes, epoch, val=True)

            test_loss, test_acc_close, test_overall, \
            test_unk, test_acc_ood, roc, close_roc \
                = test(args, test_loader, test_model, num_classes, epoch)
            log_dict = {}

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            log_dict['train/train_loss'] = losses.avg
            log_dict['test/test_acc'] = test_acc_close
            log_dict['test/test_loss'] = test_loss

            log_dict['test/test_acc_overall'] = test_overall
            log_dict['test/roc'] = roc
            log_dict['test/close_roc'] = close_roc

            wandb_logger.log(log_dict)

            is_best = val_test_acc_close >= best_acc_val
            best_acc_val = max(val_test_acc_close, best_acc_val)
            if is_best:
                overall_valid = test_overall
                close_valid = test_acc_close
                unk_valid = test_unk

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'acc overall': test_overall,
                'unk': test_unk,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            logger.info('Valid overall acc: {:.3f}'.format(overall_valid))
            logger.info('Valid unk acc: {:.3f}'.format(unk_valid))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
    if args.local_rank in [-1, 0]:
        args.writer.close()
