import logging
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, all_gather_object, is_main_process
from .modeling import PtGenerator
from .train_utils import fix_random_seed, iou
from .modeling import OnVTG

logger = logging.getLogger(__name__)

class Evaluator:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt['seed'])

        # prepare dataset
        eval_data_opt = opt['train']['data']
        eval_data_opt.update(opt['eval']['data'])
        dataset = make_dataset(eval_data_opt, is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0,
            world_size=get_world_size(), rank=get_rank()
        )

        # register model hyperparameters
        self.max_vid_len = opt['eval']['max_vid_len']
        self.vid_stride = opt['model']['vid_net']['stride']
        self.num_fpn_levels = len(opt['model']['vid_net']['memory_size'])
        self.input_vid_len = self.max_vid_len * self.vid_stride
        self.pt_gen = PtGenerator(max_seq_len=self.max_vid_len, num_fpn_levels=self.num_fpn_levels).cuda()

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.max_rank_num = opt['eval'].get('max_rank_num', self.topk)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        self.delays = np.zeros((2, len(self.iou_threshs)))
        self.text_cnt = 0
        self.eval_mode = opt['eval'].get('eval_mode', 'streaming')
        self.stream_thresholds = opt['eval'].get(
            'stream_thresholds',
            [round(x * 0.05, 2) for x in range(1, 15)]
        )
        self.cls_threshold = opt['eval'].get('cls_threshold', 0.3)
        self.score_alpha = opt['eval'].get('score_alpha', 2.0)
        self.trigger_threshold = opt['eval'].get('trigger_threshold', 0.4)
        self.stream_nms_iou = opt['eval'].get('stream_nms_iou', 0.7)

        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']
        self.future_topk = opt['eval']['future_topk']

        self.model = None
    
    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def run(self):
        assert self.model is not None
        self.model.eval()
        logger.info("Evaluation started.")
        results = []
        targets = []
        pbar = tqdm(self.dataloader) if is_main_process() else self.dataloader
        for data_list in pbar:
            res = self.predict(data_list[0])
            tgt = data_list[0]['segment'].tolist()
            assert len(res) == len(tgt)
            results += res
            targets += tgt

        all_results =  [None for _ in range(get_world_size())]
        all_gather_object(all_results, results)
        all_targets =  [None for _ in range(get_world_size())]
        all_gather_object(all_targets, targets)

        all_results = [res for results in all_results for res in results]
        all_targets = [tgt for targets in all_targets for tgt in targets]

        if self.eval_mode == 'offline':
            logger.info("eval_mode: offline")
            score = self.calc_metric(all_results, all_targets)
        elif self.eval_mode == 'streaming':
            logger.info(
                f"eval_mode: streaming, cls_threshold: {self.cls_threshold}, "
                f"score_alpha: {self.score_alpha}, trigger_threshold: {self.trigger_threshold}, "
                f"stream_nms_iou: {self.stream_nms_iou}"
            )
            score = self.calc_stream_metric(all_results, all_targets)
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")

        return score

    def calc_stream_metric(self, all_results, all_targets):
        best_score = 0.0
        best_threshold = None
        for score_thresh in self._stream_thresholds():
            self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
            self.delays = np.zeros((2, len(self.iou_threshs)))
            self.text_cnt = 0
            all_iou = []

            for result, target in zip(all_results, all_targets):
                threshold_result = result['stream'][score_thresh]
                segs = threshold_result['segments']
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)

                self.text_cnt += 1
                if segs.size(0) == 0:
                    all_iou.append(0.0)
                    continue

                segs, timestamps = segs[:, :2], segs[:, 2:]
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
                self.delays[0] += (timestamps[0, 0] - target[0, 0]).cpu().numpy() * (self.iou_threshs <= iou_topk[0].item())
                self.delays[1] += (timestamps[0, 1] - target[0, 1]).cpu().numpy() * (self.iou_threshs <= iou_topk[0].item())
                all_iou.append(iou_topk[0].item())

            miou = sum(all_iou) / max(self.text_cnt, 1)
            rank1_iou50 = self.log_stream(score_thresh, miou)
            if rank1_iou50 > best_score:
                best_score = rank1_iou50
                best_threshold = score_thresh

        logger.info(f"\nBest streaming threshold: {best_threshold:.2f}, Rank@1 IoU@0.5: {(best_score * 100):.2f}")
        return best_score

    def calc_metric(self, all_results, all_targets, use_future=False):
        all_iou = []
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        self.delays = np.zeros((2, len(self.iou_threshs)))
        self.text_cnt = 0
        for result, target in zip(all_results, all_targets):
            if use_future:
                segs, scores = result['future_segments'], result['future_scores']
            else:
                segs, scores = result['segments'], result['scores']
            idx = scores.argsort(descending=True)
            segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
            target = torch.as_tensor(target, dtype=torch.float)
            target = target.expand(len(segs), -1)
            
            if segs.size(0) == 0:
                continue
            segs, timestamps = segs[:, :2], segs[:, 2:]
            iou_topk = iou(segs, target)
            iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
            self.counts += (iou_n[:, None] >= self.iou_threshs[None])
            self.delays[0] += (timestamps[0, 0] - target[0, 0]).cpu().numpy() * (self.iou_threshs <= iou_topk[0].item())
            self.delays[1] += (timestamps[0, 1] - target[0, 1]).cpu().numpy() * (self.iou_threshs <= iou_topk[0].item())
            all_iou.append(iou_topk[0].item())
            self.text_cnt += 1
        
        self.log(use_future)
        miou = sum(all_iou) / max(self.text_cnt, 1)
        
        return miou

    def predict(self, data):
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid'][0]
        vid_len = vid.size(-1)
        min_vid_lens = 2 ** (self.num_fpn_levels - 1) * self.vid_stride
        input_vid_len = math.ceil(vid_len / min_vid_lens) * min_vid_lens 
        
        vid = F.pad(vid, (0, input_vid_len - vid_len)).view(1, 1, -1, input_vid_len)
        mask = torch.arange(input_vid_len).view(1, 1, -1) < vid_len
        vid = vid.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        
        fpn, fpn_masks = self.model.encode_video(vid, mask)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        fpn_logits_list, fpn_offsets_list, fpn_conf_logits_list = tuple(), tuple(), tuple()
        for text, text_mask in zip(text_list, text_mask_list):
            fpn_logits, fpn_offsets, fpn_conf_logits, _, _, _, _ = \
                self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask, text_size=text.size(0))
            fpn_logits_list += (tuple(f.squeeze(1) for f in fpn_logits), )
            fpn_offsets_list += (tuple(f.squeeze(1) for f in fpn_offsets), )
            fpn_conf_logits_list += (tuple(f.squeeze(1) for f in fpn_conf_logits), )
        
        fpn_masks = [m.squeeze(1).squeeze(1) for m in fpn_masks]
        # collect segments and their scores
        results_list = tuple()
        for fpn_logits, fpn_offsets, fpn_conf_logits in zip(
            fpn_logits_list, fpn_offsets_list, fpn_conf_logits_list
        ):
            if self.eval_mode == 'offline':
                segs, scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, None, None
                )
                results_list += ({'segments': segs, 'scores': scores}, )
            else:
                stream_results = self._collect_streaming_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_conf_logits, fpn_masks
                )
                results_list += ({'stream': stream_results}, )

        results = []
        for result in results_list:
            # convert segments to timestamps in seconds
            clip_stride = data['clip_stride']
            clip_size = data['clip_size']
            fps = data['fps']
            duration = data['duration']

            if self.eval_mode == 'offline':
                segs = result['segments']
                scores = result['scores']
                if len(segs) > 0:
                    segs = segs * self.vid_stride
                    segs = (segs * clip_stride + 0.5 * clip_size) / fps
                    segs = torch.clamp(segs, min=0, max=duration)
                results.append({'segments': segs.cpu(), 'scores': scores.cpu()})
            else:
                converted = {}
                for score_thresh, threshold_result in result['stream'].items():
                    segs = threshold_result['segments']
                    scores = threshold_result['scores']
                    if len(segs) > 0:
                        segs = segs * self.vid_stride
                        segs = (segs * clip_stride + 0.5 * clip_size) / fps
                        segs = torch.clamp(segs, min=0, max=duration)
                    converted[score_thresh] = {
                        'segments': segs.cpu(),
                        'scores': scores.cpu(),
                    }

                if is_main_process():
                    threshold = self._stream_thresholds()[0]
                    sample = converted[threshold]['segments'][:3]
                    logger.info(
                        f"stream triggers: {len(converted[threshold]['segments'])}, "
                        f"threshold: {threshold}, first_segments: {sample.tolist()}"
                    )
                results.append({'stream': converted})

        return results

    def _stream_thresholds(self):
        if self.stream_thresholds is None:
            return [self.trigger_threshold]
        return list(self.stream_thresholds)

    def compute_stream_score(self, cls_score, conf_score):
        # cls-gated quality score: conf predicts localization quality, not background.
        return torch.where(
            cls_score >= self.cls_threshold,
            (cls_score ** self.score_alpha) * conf_score,
            torch.zeros_like(cls_score),
        )

    def _collect_streaming_segments(self, fpn_points, fpn_logits, fpn_offsets, fpn_conf_logits, fpn_masks):
        thresholds = self._stream_thresholds()
        selected_segs = {threshold: [] for threshold in thresholds}
        selected_scores = {threshold: [] for threshold in thresholds}
        events = []

        for level, (points, logits, offsets, conf_logits, masks) in enumerate(zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_conf_logits, fpn_masks
        )):
            logits, offsets, conf_logits, masks = logits[0], offsets[0], conf_logits[0], masks[0]
            cls_scores = torch.sigmoid(logits)
            conf_scores = torch.sigmoid(conf_logits)
            scores = self.compute_stream_score(cls_scores, conf_scores) * masks.float()

            pt_ctr = points[:, 0]
            left = pt_ctr - offsets[:, 0] * points[:, 3]
            right = pt_ctr + offsets[:, 1] * points[:, 3]
            timestamp = pt_ctr
            segs = torch.stack((left, right, timestamp, timestamp), dim=-1)

            valid = torch.logical_and(masks, (right - left) > self.seg_len_thresh)
            idxs = torch.nonzero(valid, as_tuple=False).flatten()
            for idx in idxs:
                events.append((
                    float(timestamp[idx].item()),
                    level,
                    float(points[idx, 0].item()),
                    segs[idx],
                    scores[idx],
                    cls_scores[idx],
                    conf_scores[idx],
                ))

        # This is the streaming order: as soon as a point becomes available.
        # All hierarchy levels participate in the same time-ordered stream.
        events.sort(key=lambda x: (x[0], x[1], x[2]))

        for point_time, level, _, seg, score, cls_score, conf_score in events:
            score_value = float(score.item())
            for threshold in thresholds:
                if len(selected_segs[threshold]) >= self.max_rank_num:
                    continue
                if score_value < threshold:
                    continue
                if self._is_stream_duplicate(seg[:2], selected_segs[threshold]):
                    continue
                selected_segs[threshold].append(seg)
                selected_scores[threshold].append(score)

        results = {}
        device = fpn_points[0].device
        for threshold in thresholds:
            if selected_segs[threshold]:
                segs = torch.stack(selected_segs[threshold])
                scores = torch.stack(selected_scores[threshold])
            else:
                segs = torch.empty((0, 4), dtype=fpn_points[0].dtype, device=device)
                scores = torch.empty((0,), dtype=fpn_points[0].dtype, device=device)
            results[threshold] = {'segments': segs, 'scores': scores}

        return results

    def _is_stream_duplicate(self, seg, selected):
        if not selected:
            return False

        selected = torch.stack([x[:2] for x in selected])
        start = torch.maximum(seg[0], selected[:, 0])
        end = torch.minimum(seg[1], selected[:, 1])
        inter = (end - start).clamp(min=0)
        union = (
            (seg[1] - seg[0])
            + (selected[:, 1] - selected[:, 0])
            - inter
        ).clamp(min=1e-6)
        return bool((inter / union >= self.stream_nms_iou).any().item())
    

    def _collect_segments(
        self,
        fpn_points,
        fpn_logits,
        fpn_offsets,
        fpn_masks,
        future_fpn_logits, 
        future_fpn_offsets
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            scores = torch.sigmoid(logits)
            scores *= masks.float()

            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        if not scores_list:
            device = fpn_points[0].device
            return (
                torch.empty((0, 4), dtype=fpn_points[0].dtype, device=device),
                torch.empty((0,), dtype=fpn_points[0].dtype, device=device),
            )
        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)


        if future_fpn_logits is not None:
            future_masks = fpn_masks[0].squeeze(0)
            future_points = fpn_points[0]
            future_scores = future_fpn_logits[0].squeeze(0).sigmoid() * future_masks
            future_offsets = future_fpn_offsets[0].squeeze(0).squeeze(-1)

            n_topk = min(len(future_points), self.future_topk)
            future_idx = future_scores.argsort(descending=True)[:n_topk]
            future_scores = future_scores[future_idx]
            future_start = future_points[future_idx, 0] - future_offsets[future_idx]
            future_timestamp = future_points[future_idx, 0] + future_points[future_idx, 3]

        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        timestamp = pt_ctr + points[:, 3]
        segs = torch.stack((left, right, timestamp, timestamp), dim=-1)

        seg_lens = segs[:, 1] - segs[:, 0]
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]


        return segs, scores

    def log(self, use_future):
        log_str = "\nuse future prediction" if use_future else "\nno future prediction"
        metrics = self.counts / self.text_cnt
        delay = self.delays / self.counts[:1]
        log_str += "\n----"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        log_str += "\n----"
        for j, thresh in enumerate(self.iou_threshs):
            log_str += (
                f"\nDelay, IoU@{thresh:.1f}: "
                f"{(delay[0, j]):.2f}, {(delay[1, j]):.2f}"
            )
        log_str += (
            f"\nDelay, Avg: "
            f"{(delay[0].mean()):.2f}, {(delay[1].mean()):.2f}"
        )
        logger.info(log_str)

        return metrics

    def log_stream(self, score_thresh, miou):
        metrics = self.counts / max(self.text_cnt, 1)
        delay = np.divide(
            self.delays,
            self.counts[:1],
            out=np.zeros_like(self.delays),
            where=self.counts[:1] != 0,
        )
        log_str = (
            f"\nStreaming trigger_threshold: {score_thresh:.2f}"
            f"\ncls_threshold: {self.cls_threshold:.2f}, score_alpha: {self.score_alpha:.2f}, "
            f"stream_nms_iou: {self.stream_nms_iou:.2f}"
        )
        log_str += "\n----"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nStreaming R@{rank}, IoU={thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        log_str += "\n----"
        for j, thresh in enumerate(self.iou_threshs):
            log_str += (
                f"\nDelay, IoU@{thresh:.1f}: "
                f"{(delay[0, j]):.2f}, {(delay[1, j]):.2f}"
            )
        log_str += f"\nTop1 mIoU: {(miou * 100):.2f}"
        logger.info(log_str)

        iou50_idx = np.where(np.isclose(self.iou_threshs, 0.5))[0]
        if len(iou50_idx) == 0:
            return metrics[0].mean()
        return metrics[0, iou50_idx[0]]

    def nms(self, segs, scores, thresh=0.3, topk=5):
        selected_idxs = []
        
        idxs = torch.arange(scores.size(0))
        
        while idxs.numel() > 0 and len(selected_idxs) < topk:
            i = idxs[0].item()
            selected_idxs.append(i)
            
            if len(selected_idxs) == topk:
                break

            start_i, end_i = segs[i, :2]
            remaining_segs = segs[idxs[1:], :2]
            start_rem = remaining_segs[:, 0]
            end_rem = remaining_segs[:, 1]
            
            inter_start = torch.max(start_i, start_rem)
            inter_end = torch.min(end_i, end_rem)
            inter_len = (inter_end - inter_start).clamp(min=0)
            union_len = (end_i - start_i) + (end_rem - start_rem) - inter_len
            iou = inter_len / union_len
            
            keep = iou <= thresh
            idxs = idxs[1:][keep]
        
        if len(selected_idxs) > 0:
            selected_idxs = torch.tensor(selected_idxs)
            segs, scores = segs[selected_idxs], scores[selected_idxs]
    
        return segs, scores
    
    def load(self, model_path):
        if self.model is None:
            self.model = OnVTG(self.opt['model']).cuda()
        model_ckpt = torch.load(model_path, map_location='cpu')
        missing, unexpected = self.model.load_state_dict(model_ckpt['model_ema'], strict=False)
        if missing:
            logger.info("Missing keys when loading eval checkpoint: " + ", ".join(missing))
        if unexpected:
            logger.info("Unexpected keys when loading eval checkpoint: " + ", ".join(unexpected))
        logger.info(f"Loaded checkpoint from {model_path}.")
