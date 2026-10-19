"""Adapt Ultralytics 8.4.0 YOLO26 to multiple heads.

Weights are never downloaded implicitly. Output boxes use normalized xyxy."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def require_ultralytics():
    version = importlib.metadata.version('ultralytics')
    if version != '8.4.0':
        raise RuntimeError(f'This adapter requires ultralytics==8.4.0; found {version}')


def default_prototypes():
    return np.vstack([np.ones(5) / np.sqrt(5), np.eye(5)]).tolist()


class InputFusion(nn.Module):
    """Fuse five channels to RGB, initialized as an RGB pass-through."""
    def __init__(self):
        super().__init__()
        self.residual = nn.Conv2d(5, 3, 1, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 5:
            raise ValueError('Expected Bx5xHxW normalized RGB/depth/NIR input')
        return x[:, :3] + self.residual(x)


class ExpertHead(nn.Module):
    def __init__(self, general, channels, width=1.0):
        super().__init__()
        if width == 1:
            self.projection = nn.ModuleList([nn.Identity() for _ in channels])
            self.detect = deepcopy(general)
        else:
            from ultralytics.nn.modules.head import Detect
            reduced = tuple(max(8, int(c * width) // 8 * 8) for c in channels)
            self.projection = nn.ModuleList([nn.Conv2d(c, r, 1) for c, r in zip(channels, reduced)])
            self.detect = Detect(general.nc, reg_max=general.reg_max, end2end=general.end2end, ch=reduced)
            self.detect.stride = general.stride.clone()
            self.detect.bias_init()
            # Inherit only shape-compatible general-head tensors.
            source = general.state_dict()
            self.detect.load_state_dict({k: source[k] for k, v in self.detect.state_dict().items()
                                         if k in source and source[k].shape == v.shape}, strict=False)

    def forward(self, features):
        return self.detect([layer(x) for layer, x in zip(self.projection, features)])

    def predict_raw(self, features):
        """Run only the native prediction branch."""
        projected = [layer(x) for layer, x in zip(self.projection, features)]
        branch = self.detect.one2one if self.detect.end2end else self.detect.one2many
        return self.detect.forward_head(projected, **branch)

    def prediction_parameters(self):
        branch = self.detect.one2one if self.detect.end2end else self.detect.one2many
        modules = [self.projection, self.detect.dfl, *branch.values()]
        parameters = {id(p): p for module in modules for p in module.parameters()}
        return sum(p.numel() for p in parameters.values())


class RoutedYOLO(nn.Module):
    schema_version = 2

    def __init__(self, config):
        super().__init__()
        require_ultralytics()
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG_DICT
        from ultralytics import YOLO
        cfg = config.get('model', config)
        self.config = dict(cfg)
        self.classes = int(cfg.get('classes', 2))
        if self.classes < 2:
            raise ValueError('Source-class margin requires at least two classes')
        self.specialists = int(cfg.get('specialists', 5))
        if self.specialists not in (0, 5):
            raise ValueError('Use five specialists for TE-Fed, or zero for general-only baselines')
        architecture = cfg.get('architecture', 'yolo26s.yaml')
        base = DetectionModel(architecture, ch=3, nc=self.classes, verbose=False)
        weights = cfg.get('weights', '')
        if weights:
            if not Path(weights).is_file():
                raise FileNotFoundError(weights)
            base.load(YOLO(weights).model, verbose=False)
        self.fusion = InputFusion()
        self.trunk = nn.ModuleList(list(base.model)[:-1])
        general = base.model[-1]
        self.feature_indices = tuple(general.f)
        # Read pyramid sizes without updating BN statistics.
        self.trunk.eval()
        with torch.no_grad():
            features = self._trunk_forward(torch.zeros(1, 3, 64, 64))
        channels = tuple(x.shape[1] for x in features)
        self.contrastive_feature_level = cfg.get('contrastive_feature_level', -1)
        self.contrastive_roi_size = cfg.get('contrastive_roi_size', 1)
        self.contrastive_sampling_ratio = cfg.get('contrastive_sampling_ratio', 2)
        contrastive_dim = cfg.get('contrastive_dim', 128)
        for name, value in [('contrastive_dim', contrastive_dim),
                            ('contrastive_roi_size', self.contrastive_roi_size),
                            ('contrastive_sampling_ratio', self.contrastive_sampling_ratio)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        level = self.contrastive_feature_level
        if isinstance(level, bool) or not isinstance(level, int) or not -len(channels) <= level < len(channels):
            raise ValueError('contrastive_feature_level is outside the detector pyramid')
        self.contrastive_projection = nn.Linear(channels[level] * self.contrastive_roi_size ** 2, contrastive_dim)
        width = float(cfg.get('specialist_width', 0.75))
        if not 0 < width <= 1:
            raise ValueError('specialist_width must be in (0, 1]')
        self.heads = nn.ModuleList([ExpertHead(general, channels)] +
                                   [ExpertHead(general, channels, width) for _ in range(self.specialists)])
        epochs = int(config.get('experiment',{}).get('rounds',30)) * int(config.get('experiment',{}).get('local_epochs',2))
        self.loss_args = SimpleNamespace(**{**DEFAULT_CFG_DICT, 'epochs': epochs})
        self.last_proposal_counts = {}
        self.train()

    def _trunk_forward(self, x):
        outputs = []
        for layer in self.trunk:
            source = layer.f
            if source != -1:
                x = outputs[source] if isinstance(source, int) else [x if j == -1 else outputs[j] for j in source]
            x = layer(x)
            outputs.append(x)
        return [outputs[i] for i in self.feature_indices]

    def features(self, x):
        return self._trunk_forward(self.fusion(x))

    def forward(self, x, active_heads=(0,)):
        features = self.features(x)
        return {int(m): self.heads[int(m)](features) for m in sorted(set(active_heads))}

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, 'heads'):
            for head in self.heads:
                for name in ['stride', 'anchors', 'strides']:
                    value = getattr(head.detect, name, None)
                    if isinstance(value, torch.Tensor):
                        setattr(head.detect, name, fn(value))
        return self

    def make_criterion(self, head_id):
        from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
        # Expose head, stride, and hyperparameters to the native loss.
        loss_model = nn.Module()
        loss_model.model = nn.ModuleList([self.heads[head_id].detect])
        loss_model.args = self.loss_args
        return E2ELoss(loss_model) if self.heads[head_id].detect.end2end else v8DetectionLoss(loss_model)

    def decode_head(self, output, head_id, image_size, *, branch='one2many'):
        """Decode native distances without detaching strong-view gradients."""
        from ultralytics.utils.tal import make_anchors, dist2bbox
        raw = output[1] if isinstance(output, tuple) else output
        raw = raw.get(branch, raw)
        logits = raw['scores'].permute(0, 2, 1).float()
        distance = raw['boxes'].permute(0, 2, 1).float()
        head = self.heads[int(head_id)].detect
        if head.reg_max > 1:
            batch, anchors, _ = distance.shape
            distance = distance.reshape(batch, anchors, 4, head.reg_max).softmax(-1).matmul(
                torch.arange(head.reg_max, device=distance.device, dtype=torch.float))
        anchors, stride = make_anchors(raw['feats'], head.stride, 0.5)
        boxes = dist2bbox(distance, anchors, xywh=False) * stride
        height, width = image_size
        boxes = boxes / boxes.new_tensor([width, height, width, height])
        return {'boxes': boxes.clamp(0, 1), 'logits': logits,
                'scores': logits.sigmoid().amax(-1)}

    def project_regions(self, feature, rois):
        """Project RoIAlign features; boxes use normalized xyxy coordinates."""
        from torchvision.ops import roi_align
        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError('RoIs must contain image index and normalized xyxy box')
        height, width = feature.shape[-2:]
        scaled = rois.float() * rois.new_tensor([1, width, height, width, height])
        pooled = roi_align(feature.float(), scaled, output_size=self.contrastive_roi_size,
                           spatial_scale=1., sampling_ratio=self.contrastive_sampling_ratio, aligned=True)
        return self.contrastive_projection(pooled.flatten(1))

    def _contrastive_features(self, view):
        """Keep fused-input BN statistics during modality-specific forwards."""
        normalizers = [(layer, layer.training) for layer in self.trunk.modules()
                       if isinstance(layer, nn.modules.batchnorm._BatchNorm)]
        try:
            for layer, _ in normalizers:
                layer.eval()
            return self.features(view)[self.contrastive_feature_level]
        finally:
            for layer, training in normalizers:
                layer.train(training)

    def contrastive_regions(self, x, targets, availability):
        """Return candidate-aligned modality projections for one local minibatch."""
        if len(targets) != x.shape[0] or tuple(availability.shape) != (x.shape[0], 3):
            raise ValueError('Contrastive inputs require targets and three modality flags per image')
        availability = availability.to(device=x.device, dtype=torch.bool)
        counts = [len(items) for items in targets]
        total = sum(counts)
        candidate_flags = torch.repeat_interleave(availability, torch.tensor(counts, device=x.device), dim=0)
        offsets = np.cumsum([0, *counts]).tolist()
        views = []
        for modality in range(3):
            images = [i for i, count in enumerate(counts) if count and bool(availability[i, modality])]
            projected = x.new_zeros((total, self.contrastive_projection.out_features), dtype=torch.float32)
            if images:
                source = x[images]
                view = torch.zeros_like(source)
                if modality == 0:
                    view[:, :3] = source[:, :3]
                else:
                    view[:, modality + 2] = source[:, modality + 2]
                feature = self._contrastive_features(view)
                rois = [[local, *target['box']] for local, image in enumerate(images) for target in targets[image]]
                indices = [j for image in images for j in range(offsets[image], offsets[image + 1])]
                values = self.project_regions(feature, torch.tensor(rois, device=x.device, dtype=torch.float32))
                projected = projected.index_copy(0, torch.tensor(indices, device=x.device), values.float())
            views.append(projected)
        return views, candidate_flags

    @torch.no_grad()
    def predict_heads(self, x, active_heads=(0,), *, confidence=0.01, max_detections=100,
                      temperature=1.0, cached_features=None):
        if temperature <= 0:
            raise ValueError('Calibration temperature must be positive')
        was_training = self.training
        self.eval()
        try:
            features = cached_features if cached_features is not None else self.features(x)
            result = {}
            for m in sorted(set(active_heads)):
                head = self.heads[int(m)]
                output = head.predict_raw(features)
                decoded = self.decode_head(output, m, x.shape[-2:], branch='one2one')
                logits, boxes = decoded['logits'], decoded['boxes']
                posterior = (logits / temperature).softmax(-1)
                detector_scores = decoded['scores']
                self.last_proposal_counts[int(m)] = (detector_scores >= .05).sum(-1).cpu().tolist()
                batch = []
                for b in range(x.shape[0]):
                    inds = (detector_scores[b] >= confidence).nonzero().flatten()
                    inds = inds[detector_scores[b, inds].argsort(descending=True, stable=True)[:max_detections]]
                    detections = []
                    for idx in inds.tolist():
                        box = boxes[b, idx].clamp(0, 1).tolist()
                        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                            continue
                        probs = posterior[b, idx].tolist()
                        detections.append({'box': box, 'posterior': probs, 'class_id': int(np.argmax(probs)),
                                           'score': float(detector_scores[b, idx]), 'index': idx})
                    batch.append(detections)
                result[int(m)] = batch
            return result
        finally:
            self.train(was_training)

    def export_state(self):
        backbone = {f'fusion.{k}': v.detach().cpu().numpy().copy() for k, v in self.fusion.state_dict().items()
                    if v.is_floating_point()}
        backbone.update({f'trunk.{k}': v.detach().cpu().numpy().copy() for k, v in self.trunk.state_dict().items()
                         if v.is_floating_point()})
        backbone.update({f'contrastive_projection.{k}': v.detach().cpu().numpy().copy()
                         for k, v in self.contrastive_projection.state_dict().items()})
        heads = {str(i): {k: v.detach().cpu().numpy().copy() for k, v in h.state_dict().items()
                          if v.is_floating_point()} for i, h in enumerate(self.heads)}
        return {'backbone': backbone, 'heads': heads}

    def import_state(self, state):
        sd = self.state_dict()
        updates = dict(state['backbone'])
        for i, head in state['heads'].items():
            updates.update({f'heads.{i}.{k}': v for k, v in head.items()})
        expected = {k for k,v in sd.items() if v.is_floating_point()}
        if set(updates) != expected:
            raise ValueError(f'State schema differs: {set(updates) ^ expected}')
        for key, value in updates.items():
            arr = np.asarray(value)
            if arr.shape != tuple(sd[key].shape) or not np.isfinite(arr).all():
                raise ValueError(f'Invalid model state: {key}')
            sd[key] = torch.as_tensor(arr, device=sd[key].device, dtype=sd[key].dtype)
        self.load_state_dict(sd, strict=True)

    def capacity(self, active_heads=None):
        """Count detector storage and the executed prediction parameter subset."""
        if active_heads is None:
            active_heads = tuple(range(min(3, len(self.heads))))
        active_heads = tuple(active_heads)
        if (not active_heads or any(isinstance(i, bool) or not isinstance(i, int)
                                    or not 0 <= i < len(self.heads) for i in active_heads)
                or len(set(active_heads)) != len(active_heads)):
            raise ValueError('active_heads must contain distinct valid head indices')
        shared_prediction = sum(p.numel() for module in [self.trunk, self.fusion]
                                for p in module.parameters())
        prediction_heads = [head.prediction_parameters() for head in self.heads]
        return {'total_parameters': sum(p.numel() for p in self.parameters()),
                'shared_parameters': sum(p.numel() for p in self.trunk.parameters()) +
                                     sum(p.numel() for p in self.fusion.parameters()) +
                                     sum(p.numel() for p in self.contrastive_projection.parameters()),
                'contrastive_parameters': sum(p.numel() for p in self.contrastive_projection.parameters()),
                'head_parameters': [sum(p.numel() for p in h.parameters()) for h in self.heads],
                'prediction_shared_parameters': shared_prediction,
                'prediction_head_parameters': prediction_heads,
                'active_heads': list(active_heads),
                'active_prediction_parameters': shared_prediction + sum(prediction_heads[i] for i in active_heads),
                'all_heads_prediction_parameters': shared_prediction + sum(prediction_heads),
                'scope': 'unfused detector; storage includes both native training branches and contrastive projection; '
                         'prediction uses the native prediction branch and excludes the contrastive projection; '
                         'router, EMA teacher, buffers, optimizer state and repeated cue forwards are excluded'}


class Router(nn.Module):
    """Initialize the router projection and fixed prototypes."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.encoder = nn.Linear(17, 5)
        self.temperature = float(temperature)
        self.register_buffer('prototypes', torch.tensor(default_prototypes(), dtype=torch.float))

    def forward(self, vector):
        q = F.normalize(self.encoder(vector), dim=-1, eps=1e-8)
        return (q @ self.prototypes.T / self.temperature).softmax(-1)


def sample_tensor(sample, device='cpu', *, rgb_only=False):
    rgb = np.asarray(sample['rgb'], dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    aux = []
    for key in ['depth', 'nir']:
        value = sample.get(key)
        if value is None or rgb_only:
            aux.append(np.zeros((h, w), dtype=np.float32))
        else:
            value = np.asarray(value, dtype=np.float32)
            finite = np.isfinite(value)
            if key == 'depth':
                scale = float(sample.get('depth_scale',1.0))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError('A fixed positive depth_scale is required')
                value = np.clip(value/scale,0,1)
            else:
                scale = 1.0 if np.issubdtype(np.asarray(sample[key]).dtype,np.floating) and np.nanmax(value,initial=0)<=1 else 255.0
                value = np.clip(value/scale,0,1)
            aux.append(np.where(finite, value, 0))
    x = np.concatenate([rgb, aux[0][..., None], aux[1][..., None]], axis=-1)
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
