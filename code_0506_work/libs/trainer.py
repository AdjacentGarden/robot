from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import os
import shutil
import time
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .dataset import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, is_main_process
from .modeling import (
    OnVTG, PtGenerator,
    sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler
)
from .train_utils import AverageMeter, fix_random_seed, time_str
from .evaluator import Evaluator

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt['seed'])

        # build model and EMA
        self.model = OnVTG(opt['model']).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.ema_beta = opt['train'].get('ema_beta', 0.999)
        self.train_stage = opt['train'].get('train_stage', 'base')
        self.conf_loss_weight = opt['train'].get('conf_loss_weight', 1.0)
        self.hard_neg_conf = opt['train'].get('hard_neg_conf', False)
        self.hard_neg_cls_thr = opt['train'].get('hard_neg_cls_thr', 0.3)
        self.hard_neg_ratio = opt['train'].get('hard_neg_ratio', 1.0)
        self.lambda_neg_conf = opt['train'].get('lambda_neg_conf', 0.05)
        logger.info(f"train_stage: {self.train_stage}")
        if opt['train'].get('init_ckpt'):
            self.load_model_checkpoint(
                opt['train']['init_ckpt'],
                use_ema=opt['train'].get('init_ckpt_use_ema', True),
            )
            self.model_ema.load_state_dict(self.model.state_dict())

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.warmup_epochs = opt['train']['scheduler']['warmup_epochs'] = opt['train']['warmup_epochs']
        opt['train']['scheduler']['epochs'] = opt['train']['epochs']
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self._configure_train_stage()
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.tb_writer = self.loss_meters = self.timer = None

        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()])
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['vid_net']['max_seq_len']
        self.max_text_len = opt['model']['text_net']['max_seq_len']
        self.vid_stride = opt['model']['vid_net']['stride']
        self.num_fpn_levels = len(opt['model']['vid_net']['memory_size'])
        self.input_vid_len = self.max_vid_len * self.vid_stride
        self.pt_gen = PtGenerator(max_seq_len=opt['train']['data']['long_term_window_size'], num_fpn_levels=self.num_fpn_levels).cuda()

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.future_loss_norm = opt['train']['future_loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.future_loss_weight = opt['train'].get('future_loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

        self.eval = opt['train']['eval']
        if self.eval:
            self.eval_epoch_interval = opt['train']['eval_epoch_interval']
            self.evaluator = Evaluator(opt)
            self.best_score = 0.0

    def run(self):
        logger.info("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            pbar = tqdm(self.dataloader) if is_main_process() else self.dataloader
            for data_list in pbar:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            is_best = False
            if self.eval and self.epoch % self.eval_epoch_interval == 0 and self.epoch >= self.warmup_epochs:
                self.evaluator.set_model(self.model_ema)
                stop_score = self.evaluator.run()
                logger.info("Eval: "+str(stop_score))
                if stop_score > self.best_score:
                    logger.info("Best score updated: " + str(stop_score))
                    self.best_score = stop_score
                    is_best = True

            self.checkpoint(is_best=is_best)
            barrier()
        logger.info("Training completed.")

    
    def forward_backward(self, data_list):
        cls_loss = reg_loss = conf_loss = total_loss = future_cls_loss = future_reg_loss = norm = future_norm = 0
        candidate_num = hard_neg_num = conf_target_mean = conf_pred_mean = 0
        conf_target_max = conf_pred_max = None
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            conf_loss += loss_dict['conf']
            future_cls_loss += loss_dict['future_cls']
            future_reg_loss += loss_dict['future_reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']
            future_norm += loss_dict['future_norm']
            candidate_num += loss_dict['candidate_num']
            hard_neg_num += loss_dict['hard_neg_num']
            conf_target_mean += loss_dict['conf_target_mean']
            conf_target_max = loss_dict['conf_target_max'] if conf_target_max is None else torch.maximum(conf_target_max, loss_dict['conf_target_max'])
            conf_pred_mean += loss_dict['conf_pred_mean']
            conf_pred_max = loss_dict['conf_pred_max'] if conf_pred_max is None else torch.maximum(conf_pred_max, loss_dict['conf_pred_max'])

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        all_norms = [torch.zeros_like(future_norm) for _ in range(get_world_size())]
        all_gather(all_norms, future_norm)
        self.future_loss_norm = (
            self.loss_norm_momentum * self.future_loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {
            'cls': cls_loss,
            'reg': reg_loss,
            'conf': conf_loss,
            'future_cls': future_cls_loss,
            'future_reg': future_reg_loss,
            'total': total_loss,
            'candidate_num': candidate_num,
            'hard_neg_num': hard_neg_num,
            'conf_target_mean': conf_target_mean / self.num_microbatches,
            'conf_target_max': conf_target_max,
            'conf_pred_mean': conf_pred_mean / self.num_microbatches,
            'conf_pred_max': conf_pred_max,
        }

    def _microbatch_forward_backward(self, data_list, is_last=False):
        self.model.train()
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_conf_logits, fpn_masks, future_fpn_logits, future_fpn_offsets = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_conf_logits, fpn_masks, future_fpn_logits, future_fpn_offsets = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        
        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True).flatten(0, 1)

        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        future_fpn_masks = fpn_masks[0].flatten(0, 1)
        fpn_logits = torch.cat(fpn_logits, dim=2).flatten(0, 1)
        fpn_offsets = torch.cat(fpn_offsets, dim=2).flatten(0, 1)
        fpn_conf_logits = torch.cat(fpn_conf_logits, dim=2).flatten(0, 1)
        fpn_masks = torch.cat(fpn_masks, dim=2).flatten(0, 1)
        points = torch.cat(fpn_points)
        if future_fpn_logits is not None:
            future_fpn_logits = torch.cat(future_fpn_logits, dim=2).flatten(0, 1)
        if future_fpn_offsets is not None:
            future_fpn_offsets = torch.cat(future_fpn_offsets, dim=2).flatten(0, 1)

        # annotate points
        gt_labels, gt_offsets, future_labels, future_offsets = self._annotate_points(points, targets)
        future_labels = future_labels[:, :future_fpn_masks.size(1)]
        future_offsets = future_offsets[:, :future_fpn_masks.size(1)]
        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()
        pos_future_masks = torch.logical_and(future_labels, future_fpn_masks)
        future_norm = pos_future_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()
        conf_loss, conf_stats = self._calc_conf_loss(
            conf_logits=fpn_conf_logits,
            pred_offsets=fpn_offsets,
            points=points,
            targets=targets,
            candidate_masks=pos_masks,
            valid_masks=fpn_masks,
            cls_logits=fpn_logits,
        )

        if future_fpn_logits is not None:
            future_cls_loss = self._calc_focal_loss(
                logits=future_fpn_logits[future_fpn_masks], labels=future_labels[future_fpn_masks]
            ) / self.future_loss_norm * get_world_size()
        else:
            future_cls_loss = fpn_logits.new_tensor(0.0)
        
        if future_fpn_offsets is not None:
            future_reg_loss = F.l1_loss(
                future_fpn_offsets[pos_future_masks].squeeze(-1), future_offsets[pos_future_masks], reduction='sum'
            ) / self.future_loss_norm * get_world_size()
        else:
            future_reg_loss = fpn_logits.new_tensor(0.0)

        base_loss = cls_loss + self.loss_weight * reg_loss + self.future_loss_weight * (future_reg_loss + future_cls_loss)
        if self.train_stage == 'conf_only':
            total_loss = conf_loss
        elif self.train_stage == 'finetune':
            total_loss = base_loss + self.conf_loss_weight * conf_loss
        else:
            total_loss = base_loss
        total_loss.backward()
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'conf': conf_loss.detach(),
            'future_cls': future_cls_loss.detach(),
            'future_reg': future_reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
            'future_norm': future_norm.detach(),
            'candidate_num': conf_stats['candidate_num'].detach(),
            'hard_neg_num': conf_stats['hard_neg_num'].detach(),
            'conf_target_mean': conf_stats['target_mean'].detach(),
            'conf_target_max': conf_stats['target_max'].detach(),
            'conf_pred_mean': conf_stats['pred_mean'].detach(),
            'conf_pred_max': conf_stats['pred_max'].detach(),
        }

    def _batchify_videos(self, vid_list):
        bs, nw = len(vid_list), len(vid_list[0])
        vid_list = [vid for window_list in vid_list for vid in window_list]
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        input_vid_len = max(vid_lens)
        min_vid_lens = 2 ** (self.num_fpn_levels - 1) * self.vid_stride
        input_vid_len = math.ceil(input_vid_len / min_vid_lens) * min_vid_lens
        vid = vid_list[0].new_full((bs * nw, vid_dim, input_vid_len), 0.)
        for idx in range(bs * nw):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(input_vid_len)[None] < vid_lens
        return vid.view(bs, nw, vid_dim, input_vid_len), vid_masks.view(bs, nw, input_vid_len)

    def _batchify_text(self, text_list):
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])

        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)

            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        labels_list, offsets_list, future_labels_list, future_offsets_list = tuple(), tuple(), tuple(), tuple()
        for target in targets:
            labels, offsets, future_labels, future_offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
            future_labels_list += (future_labels, )
            future_offsets_list += (future_offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        future_labels = torch.stack(future_labels_list)
        future_offsets = torch.stack(future_offsets_list)
        return labels, offsets, future_labels, future_offsets

    def _annotate_points_per_video(self, points, target):
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0] 
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        # future_offsets = target[0] - (points[:, 0] + points[:, 0])
        future_labels = torch.logical_and(pt2start < 4, pt2start > -4)

        return labels, offsets, future_labels, pt2start

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _calc_conf_loss(
        self,
        conf_logits,
        pred_offsets,
        points,
        targets,
        candidate_masks,
        valid_masks,
        cls_logits,
    ):
        # IoU-aware conf target: current reg prediction is detached so conf loss
        # trains only localization quality scoring, not the reg head.
        pred_offsets = pred_offsets.detach()
        pt_ctr = points[:, 0][None]
        pt_stride = points[:, 3][None]
        pred_left = pt_ctr - pred_offsets[..., 0] * pt_stride
        pred_right = pt_ctr + pred_offsets[..., 1] * pt_stride
        gt_left = targets[:, None, 0]
        gt_right = targets[:, None, 1]

        inter_left = torch.maximum(pred_left, gt_left)
        inter_right = torch.minimum(pred_right, gt_right)
        inter = (inter_right - inter_left).clamp(min=0)
        union = (
            (pred_right - pred_left).clamp(min=0)
            + (gt_right - gt_left).clamp(min=0)
            - inter
        ).clamp(min=1e-6)
        conf_target = (inter / union).clamp(min=0.0, max=1.0)

        pos_mask = candidate_masks
        pred_conf = torch.sigmoid(conf_logits.detach())
        zero = conf_logits.sum() * 0.0
        num_pos = pos_mask.sum()
        if num_pos > 0:
            pos_loss = F.binary_cross_entropy_with_logits(
                conf_logits[pos_mask], conf_target[pos_mask], reduction='sum'
            )
            target_values = conf_target[pos_mask]
            pred_values = pred_conf[pos_mask]
        else:
            pos_loss = zero
            target_values = conf_target.new_zeros((0,))
            pred_values = pred_conf.new_zeros((0,))

        neg_loss = zero
        hard_neg_num = conf_logits.new_tensor(0.0)
        if self.hard_neg_conf and num_pos > 0:
            cls_score = torch.sigmoid(cls_logits.detach())
            neg_pool = torch.logical_and(valid_masks, torch.logical_not(pos_mask))
            neg_pool = torch.logical_and(neg_pool, cls_score > self.hard_neg_cls_thr)
            neg_idxs = torch.nonzero(neg_pool.flatten(), as_tuple=False).flatten()
            max_neg = int(num_pos.item() * self.hard_neg_ratio)
            num_neg = min(max_neg, int(neg_idxs.numel()))
            if num_neg > 0:
                flat_scores = cls_score.flatten()[neg_idxs]
                topk = torch.topk(flat_scores, k=num_neg).indices
                sampled = neg_idxs[topk]
                flat_conf_logits = conf_logits.flatten()
                neg_loss = F.binary_cross_entropy_with_logits(
                    flat_conf_logits[sampled],
                    flat_conf_logits.new_zeros((num_neg,)),
                    reduction='sum',
                )
                hard_neg_num = conf_logits.new_tensor(float(num_neg))

        conf_loss = (
            pos_loss + self.lambda_neg_conf * neg_loss
        ) / self.loss_norm * get_world_size()
        stats = {
            'candidate_num': conf_logits.new_tensor(float(num_pos.item())),
            'hard_neg_num': hard_neg_num,
            'target_mean': target_values.mean() if target_values.numel() > 0 else conf_logits.new_tensor(0.0),
            'target_max': target_values.max() if target_values.numel() > 0 else conf_logits.new_tensor(0.0),
            'pred_mean': pred_values.mean() if pred_values.numel() > 0 else conf_logits.new_tensor(0.0),
            'pred_max': pred_values.max() if pred_values.numel() > 0 else conf_logits.new_tensor(0.0),
        }
        return conf_loss, stats

    def _configure_train_stage(self):
        if self.train_stage == 'base':
            for p in self.model.parameters():
                p.requires_grad = True
            logger.info("Base stage: all model parameters are trainable; conf loss is logged but not added to total loss.")
        elif self.train_stage == 'conf_only':
            conf_param_names = []
            for name, p in self.model.named_parameters():
                p.requires_grad = 'conf_head' in name
                if p.requires_grad:
                    conf_param_names.append(name)
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen = total - trainable
            logger.info(f"conf_only stage parameter counts: total={total}, trainable={trainable}, frozen={frozen}")
            logger.info("conf head trainable parameters:\n" + "\n".join(conf_param_names))
        elif self.train_stage == 'finetune':
            for p in self.model.parameters():
                p.requires_grad = True
            logger.info("Finetune stage: all model parameters are trainable.")
        else:
            raise ValueError(f"Unsupported train_stage: {self.train_stage}")

    def load_model_checkpoint(self, model_path, use_ema=True):
        ckpt = torch.load(model_path, map_location='cpu')
        key = 'model_ema' if use_ema and 'model_ema' in ckpt else 'model'
        missing, unexpected = self.model.load_state_dict(ckpt[key], strict=False)
        logger.info(f"Loaded init checkpoint from {model_path} using key={key}.")
        if missing:
            logger.info("Missing keys when loading init checkpoint: " + ", ".join(missing))
        if unexpected:
            logger.info("Unexpected keys when loading init checkpoint: " + ", ".join(unexpected))

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'], strict=False)
        self.model_ema.load_state_dict(model_ckpt['model_ema'], strict=False)
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        logger.info(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self, is_best=False):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        logger.info(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        Path(model_dir).mkdir(exist_ok=True)
        Path(state_dir).mkdir(exist_ok=True)
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.train_stage in ('conf_only', 'finetune'):
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"checkpoint_{self.train_stage}.pth")
            )
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )
        if is_best:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, 'best.pth')
            )
            shutil.copyfile(
                os.path.join(state_dir, 'last.pth'),
                os.path.join(state_dir, 'best.pth')
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        logger.info(log_str)
        self.tb_writer.flush()
