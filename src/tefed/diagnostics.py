"""Measure renderer-reference agreement; renderer classes are not independent labels."""
from __future__ import annotations

import numpy as np

from . import core


def renderer_target_observation(outputs, route, weights, target, *, scope):
    """Measure focal-target agreement once before local training.

    Associate by geometry, then fuse the recorded route. Missing responses
    count as failures; training and retention gates are unchanged."""
    route, weights = core._route(route, weights)
    matched = [core.associate_hypothesis(core._outputs_get(outputs, head), target['box']) for head in route]
    mass = sum(weight for weight, prediction in zip(weights, matched) if prediction is not None)
    prediction_box = prediction_class = None
    overlap = 0.0
    if mass > 0:
        prediction_box = np.clip(sum(weight * np.asarray(prediction['box'])
            for weight, prediction in zip(weights, matched) if prediction is not None) / mass, 0., 1.).tolist()
        posterior = sum(weight * np.asarray(prediction['posterior'])
            for weight, prediction in zip(weights, matched) if prediction is not None) / mass
        prediction_class = int(np.argmax(posterior))
        overlap = core.iou(prediction_box, target['box'])
    return {'predicted_class': prediction_class, 'predicted_box': prediction_box,
            'renderer_class': int(target['class_id']), 'renderer_box': list(target['box']),
            'iou': float(overlap), 'agreement': bool(prediction_class == target['class_id'] and overlap >= .5),
            'scope': scope, 'reference': 'renderer focal annotation; not independent ground truth'}
