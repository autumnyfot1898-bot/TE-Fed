"""Weight native YOLO loss per target, once per image and head.

Retain native assignment and normalization; weight foreground terms by target
reliability and background classification by mean reliability."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class _WeightContext:
    def __init__(self):
        self.target_weights = None
        self.anchor_weights = None
        self.background_weight = None


class _AssignedWeights:
    def __init__(self, native, context):
        self.native, self.context = native, context

    def __call__(self, *args, **kwargs):
        outputs = self.native(*args, **kwargs)
        foreground, gt_index = outputs[3], outputs[4]
        ctx = self.context
        w = torch.full_like(foreground, ctx.background_weight, dtype=torch.float32)
        if ctx.target_weights.shape[1] and foreground.any():
            gathered = ctx.target_weights.gather(1, gt_index.long().clamp(0, ctx.target_weights.shape[1]-1))
            w = torch.where(foreground, gathered, w)
        ctx.anchor_weights = w
        return outputs


class _WeightedBCE:
    def __init__(self, context):
        self.context = context

    def __call__(self, predictions, targets):
        return F.binary_cross_entropy_with_logits(predictions, targets, reduction='none') * self.context.anchor_weights.unsqueeze(-1)


class _WeightedBoxes:
    def __init__(self, native, context):
        self.native, self.context = native, context

    def __call__(self, *args, **kwargs):
        args = list(args)
        # Weight positive localization; keep the native target_scores_sum normalizer.
        args[4] = args[4] * self.context.anchor_weights.unsqueeze(-1)
        return self.native(*args, **kwargs)


def target_batch(targets, device):
    boxes = torch.tensor([t['box'] for t in targets], device=device, dtype=torch.float32).reshape(-1, 4)
    xywh = boxes.clone()
    if boxes.numel():
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    return {'batch_idx': torch.zeros(len(targets), device=device),
            'cls': torch.tensor([t['class_id'] for t in targets], device=device).float().reshape(-1, 1),
            'bboxes': xywh}


class WeightedNativeLoss:
    def __init__(self, model, head_id):
        self.native = model.make_criterion(head_id)
        self.contexts = []
        inners = [self.native.one2many, self.native.one2one] if hasattr(self.native, 'one2many') else [self.native]
        for inner in inners:
            context = _WeightContext()
            inner.assigner = _AssignedWeights(inner.assigner, context)
            inner.bce = _WeightedBCE(context)
            inner.bbox_loss = _WeightedBoxes(inner.bbox_loss, context)
            self.contexts.append(context)

    def __call__(self, predictions, targets, weights, device):
        if len(targets) != len(weights):
            raise ValueError('Each target must have one reliability/route weight')
        w = torch.as_tensor(weights, device=device, dtype=torch.float32)
        if not torch.isfinite(w).all() or (w < 0).any() or (w > 1).any():
            raise ValueError('Target weights must be finite and in [0,1]')
        for context in self.contexts:
            context.target_weights = w.reshape(1, -1)
            # Empty supervised frames are negatives; the assembler excludes empty pseudo sets.
            context.background_weight = float(w.mean()) if w.numel() else 1.0
            context.anchor_weights = None
        loss, components = self.native(predictions, target_batch(targets, device))
        return loss.float().sum(), components

    def epoch_update(self):
        if hasattr(self.native, 'update'):
            self.native.update()


def contrastive_alignment(embeddings, temperature=0.1, *, availability=None):
    """Average InfoNCE over all available ordered modality pairs in a minibatch."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Contrastive temperature must be finite and positive')
    if not embeddings:
        return torch.zeros(())
    shape = embeddings[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in embeddings):
        raise ValueError('Contrastive modalities must share candidate order and embedding dimension')
    views = [value.float() / (value.float().norm(dim=-1, keepdim=True) + 1e-8) for value in embeddings]
    flags = torch.ones((shape[0], len(views)), device=views[0].device, dtype=torch.bool) if availability is None else availability.to(device=views[0].device, dtype=torch.bool)
    if tuple(flags.shape) != (shape[0], len(views)):
        raise ValueError('Contrastive availability requires one flag per candidate and modality')
    total = sum(value.sum() * 0 for value in views)
    pairs = 0
    for a, query_view in enumerate(views):
        for b, key_view in enumerate(views):
            if a == b:
                continue
            queries = (flags[:, a] & flags[:, b]).nonzero().flatten()
            if not len(queries):
                continue
            keys = flags[:, b].nonzero().flatten()
            logits = query_view[queries] @ key_view[keys].T / temperature
            labels = torch.searchsorted(keys, queries)
            total = total + F.cross_entropy(logits, labels, reduction='sum')
            pairs += len(queries)
    return total / pairs if pairs else total


def _aligned_giou(left, right):
    intersection = (torch.minimum(left[:, 2:], right[:, 2:]) - torch.maximum(left[:, :2], right[:, :2])).clamp_min(0).prod(-1)
    union = (left[:, 2:] - left[:, :2]).prod(-1) + (right[:, 2:] - right[:, :2]).prod(-1) - intersection
    enclosing = (torch.maximum(left[:, 2:], right[:, 2:]) - torch.minimum(left[:, :2], right[:, :2])).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-8) - (enclosing - union) / enclosing.clamp_min(1e-8)


def soft_consistency(weak, strong, *, confidence=0.01, max_detections=100, temperature=1.0,
                     return_details=False):
    """Match same-class boxes at IoU >= .50, then average half KL and half GIoU loss.

    Strong boxes must already use weak-view coordinates. Weak targets are detached.
    return_details provides the sum and pair count for minibatch normalization."""
    from torchvision.ops import box_iou
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Consistency confidence must be in [0,1]')
    if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections < 1:
        raise ValueError('Consistency max_detections must be a positive integer')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Consistency temperature must be finite and positive')
    wp, sp = weak['logits'].detach().float(), strong['logits'].float()
    wb, sb = weak['boxes'].detach().float(), strong['boxes'].float()
    if wp.ndim != 3 or sp.ndim != 3 or wp.shape[0] != sp.shape[0] or wp.shape[-1] != sp.shape[-1]:
        raise ValueError('Consistency requires batch-aligned class logits')
    if tuple(wb.shape) != (*wp.shape[:2], 4) or tuple(sb.shape) != (*sp.shape[:2], 4):
        raise ValueError('Consistency boxes must be decoded BxNx4 xyxy tensors')
    total = (sp.sum() + sb.sum()) * 0
    count = 0
    for image in range(wp.shape[0]):
        indices = []
        for boxes, logits, scores in ((wb[image], wp[image], weak['scores'][image].detach()),
                                      (sb[image].detach(), sp[image].detach(), strong['scores'][image].detach())):
            valid = torch.isfinite(boxes).all(-1) & torch.isfinite(logits).all(-1) & torch.isfinite(scores)
            valid &= (boxes[:, 2:] > boxes[:, :2]).all(-1) & (scores >= confidence)
            chosen = valid.nonzero().flatten()
            chosen = chosen[scores[chosen].argsort(descending=True, stable=True)[:max_detections]]
            indices.append(chosen)
        wi, si = indices
        if not len(wi) or not len(si):
            continue
        overlap = box_iou(wb[image, wi], sb[image, si].detach())
        same_class = wp[image, wi].argmax(-1)[:, None] == sp[image, si].detach().argmax(-1)[None, :]
        eligible = ((overlap >= .5) & same_class).nonzero()
        order = overlap[eligible[:, 0], eligible[:, 1]].argsort(descending=True, stable=True)
        used_w, used_s, pairs = set(), set(), []
        for left, right in eligible[order].tolist():
            if left not in used_w and right not in used_s:
                pairs.append((left, right)); used_w.add(left); used_s.add(right)
        if not pairs:
            continue
        matched = torch.tensor(pairs, device=sp.device)
        left, right = wi[matched[:, 0]], si[matched[:, 1]]
        posterior = (wp[image, left] / temperature).softmax(-1)
        kl = F.kl_div((sp[image, right] / temperature).log_softmax(-1), posterior, reduction='none').sum(-1)
        giou = _aligned_giou(wb[image, left], sb[image, right])
        total = total + .5 * (kl + 1 - giou).sum()
        count += len(pairs)
    return (total, count) if return_details else (total / count if count else total)
