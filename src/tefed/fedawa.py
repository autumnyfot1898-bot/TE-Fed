"""Optimize aggregation weights with FedAWA's paper objective (Eq. 3)."""
from __future__ import annotations

import math

import numpy as np


def flatten_parameters(state, parameter_names):
    """Flatten exported detector parameters, excluding running buffers."""
    values = dict(state['backbone'])
    for head, tensors in state['heads'].items():
        values.update({f'heads.{head}.{key}': value for key, value in tensors.items()})
    names = sorted(parameter_names)
    if not names or any(name not in values for name in names):
        raise ValueError('The export does not contain every detector parameter')
    return np.concatenate([np.asarray(values[name], dtype=np.float64).ravel() for name in names])


def _coordinates(vectors):
    """Preserve inner products in a small client-space representation."""
    gram = vectors @ vectors.T
    if not np.isfinite(gram).all():
        raise ValueError('FedAWA parameter inner products overflowed')
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(np.float64).eps * len(vectors) * max(float(eigenvalues[-1]), 0) * 8
    eigenvalues[eigenvalues < tolerance] = 0
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))


def paper_aggregation_weights(client_vectors, global_vector, *, logits=None, sample_counts=None,
                              steps=1, learning_rate=.001):
    """Return post-optimization weights and resumable logits, without a weight cap.

    Eq. 3 compares the aggregated parameter vector with the broadcast vector.
    The released implementation instead averages individual cosine penalties
    and returns probabilities from before its last step. This mode uses Eq. 3
    and post-step probabilities. Adam settings follow the released optimizer.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('FedAWA steps must be a positive integer')
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('FedAWA learning_rate must be finite and positive')
    reference = np.asarray(global_vector, dtype=np.float64)
    vectors = [np.asarray(vector, dtype=np.float64) for vector in client_vectors]
    if (reference.ndim != 1 or not reference.size or not np.isfinite(reference).all()
            or not vectors or any(vector.shape != reference.shape or not np.isfinite(vector).all()
                                  for vector in vectors)):
        raise ValueError('FedAWA requires finite, equally sized parameter vectors')
    count = len(vectors)
    if sample_counts is None:
        fractions = np.full(count, 1 / count)
    else:
        fractions = np.asarray(sample_counts, dtype=np.float64)
        if (fractions.shape != (count,) or not np.isfinite(fractions).all()
                or (fractions < 0).any() or not np.isfinite(fractions.sum()) or fractions.sum() <= 0):
            raise ValueError('FedAWA sample_counts must have positive total mass')
        fractions = fractions / fractions.sum()
    initial = np.log(np.maximum(fractions, np.finfo(np.float64).tiny)) if logits is None else np.asarray(logits, dtype=np.float64)
    if initial.shape != (count,) or not np.isfinite(initial).all():
        raise ValueError('FedAWA logits must match the participating clients')

    parameters = np.stack(vectors)
    updates = torch.from_numpy(_coordinates(parameters - reference))
    parameter_coordinates = torch.from_numpy(_coordinates(np.vstack([parameters, reference])))
    raw = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=learning_rate, betas=(.5, .999))

    def objective():
        weights = raw.softmax(0)
        center = weights @ updates
        spread = (weights * torch.linalg.vector_norm(updates - center, dim=1)).sum()
        aggregate = weights @ parameter_coordinates[:-1]
        alignment = 1 - F.cosine_similarity(aggregate, parameter_coordinates[-1], dim=0, eps=1e-8)
        return spread + alignment

    initial_objective = float(objective().detach())
    for _ in range(steps):
        optimizer.zero_grad()
        loss = objective()
        if not torch.isfinite(loss):
            raise ValueError('FedAWA objective is not finite')
        loss.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all():
            raise ValueError('FedAWA weight gradient is not finite')
        optimizer.step()
    final_objective = float(objective().detach())
    if not torch.isfinite(raw).all() or not math.isfinite(final_objective):
        raise ValueError('FedAWA optimized state is not finite')
    return {'weights': raw.detach().softmax(0).tolist(), 'logits': raw.detach().tolist(),
            'initial_objective': initial_objective, 'objective': final_objective,
            'steps': steps, 'learning_rate': float(learning_rate), 'definition': 'fedawa_paper_eq3',
            'initialization': 'sample_fraction_probabilities' if logits is None else 'persisted_logits',
            'probability_timing': 'after_final_optimizer_step',
            'parameter_scope': 'all detector parameters; running buffers excluded'}
