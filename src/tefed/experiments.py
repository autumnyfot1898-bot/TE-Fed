"""Explicit executable factorial protocols and artifact bindings."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256, file_digest
from itertools import product
import json
from pathlib import Path


FACTORIAL_VERSION = 'tefed-factorial-v1'
FACTORIAL_DEFINITIONS = {
    'A': {'on': 'State-dependent reliability qualification with the measured routed reliability weight.',
          'off': 'Fixed posterior qualification at 0.75; keep validity checks, routed reliability weights and correction eligibility.'},
    'B': {'on': 'TE-Fed source-evidence descriptors and stored-route verification.',
          'off': 'Generic HC held-out coarse priors and current-route verification; no effect when C is off.'},
    'C': {'on': 'Correction-capable training, rendering, verification, pool insertion and next-round selection.',
          'off': 'Server-MoE; disable the entire correction path, including descriptor collection.'},
}


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).expanduser().open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def artifact_bindings(config):
    experiment, model = config['experiment'], config.get('model', {})
    initial = model.get('initial_checkpoint')
    if not initial and not experiment.get('test_mode', False):
        raise ValueError('An experiment protocol requires model.initial_checkpoint')
    return {'manifest_sha256': file_hash(experiment['manifest']),
            'calibration_sha256': file_hash(experiment['calibration']),
            'initial_checkpoint_sha256': file_hash(initial) if initial else None}


def factorial_plans():
    definitions = canonical_hash(FACTORIAL_DEFINITIONS)
    jobs = []
    for a, b, c in product((0, 1), repeat=3):
        factors = {'A': a, 'B': b, 'C': c}
        method = 'server_moe' if not c else 'tefed' if b else 'generic_hc'
        patch = {'experiment': {'method': method},
                 'factorial': {'version': FACTORIAL_VERSION, 'definitions_sha256': definitions, 'factors': factors}}
        jobs.append((f'A{a}_B{b}_C{c}', patch))
    return jobs


def validate_factorial_config(config):
    protocol = config.get('factorial')
    if protocol is None:
        return None
    if config.get('communication', {}).get('mode', 'local') != 'local':
        raise ValueError('factorial requires communication.mode=local')
    if protocol.get('version') != FACTORIAL_VERSION or protocol.get('definitions_sha256') != canonical_hash(FACTORIAL_DEFINITIONS):
        raise ValueError('Unknown factorial protocol version or operation definitions')
    factors = protocol.get('factors', {})
    if set(factors) != {'A', 'B', 'C'} or any(type(bit) is not int or bit not in (0, 1) for bit in factors.values()):
        raise ValueError('Factorial levels must be integer A/B/C bits')
    expected = 'server_moe' if not factors['C'] else 'tefed' if factors['B'] else 'generic_hc'
    if config['experiment'].get('method') != expected:
        raise ValueError('Factorial method does not implement the declared B/C operations')
    if factors['C'] and not config['experiment'].get('test_mode', False):
        if not config.get('renderer', {}).get('enabled', True) or config.get('decision', {}).get('render_budget', 1152) <= 0:
            raise ValueError('C=1 requires enabled rendering and a positive rendering budget')
        if config.get('loss', {}).get('beta', .5) <= 0:
            raise ValueError('C=1 requires positive correction supervision')
    return factors


def apply_qualification_control(candidate, config):
    factors = validate_factorial_config(config)
    if factors is None or factors['A']:
        return candidate
    result = deepcopy(candidate)
    original = result.get('disposition')
    if not result.get('valid'):
        disposition = 'suppressed'
    elif max(result['posterior']) >= .75:
        disposition = 'qualified'
    else:
        decision = config.get('decision', {})
        cues = result['cues']
        eligible = result['D_E'] >= decision.get('delta_d', .35) and cues['C'] >= decision.get('delta_c', .5) and cues['G'] >= decision.get('delta_g', .5)
        disposition = 'correction' if eligible else 'suppressed'
    result['disposition'] = disposition
    result['qualification_control'] = {'mode': 'fixed_posterior', 'threshold': .75, 'original_disposition': original}
    return result


def factorial_manifest(config, seeds, output):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Factorial seeds must be distinct nonnegative integers')
    bindings = artifact_bindings(config)
    resolved = deepcopy(config)
    resolved.pop('_config_path', None)
    for key in ('seed', 'seeds', 'output', 'manifest', 'calibration'):
        resolved['experiment'].pop(key, None)
    resolved.get('model', {}).pop('initial_checkpoint', None)
    jobs = []
    for variant, patch in factorial_plans():
        for seed in seeds:
            jobs.append({'variant': variant, 'seed': seed, 'factors': patch['factorial']['factors'],
                         'patch': patch, 'bindings': bindings,
                         'output': str(Path(output) / variant / f'seed_{seed}')})
    body = {'version': FACTORIAL_VERSION, 'operations': FACTORIAL_DEFINITIONS,
            'definitions_sha256': canonical_hash(FACTORIAL_DEFINITIONS), 'seed_ids': seeds,
            'bindings': bindings, 'scope': 'software_fixture' if config['experiment'].get('test_mode') else 'data_experiment',
            'configuration_sha256': canonical_hash(resolved),
            'structural_null': 'B has no training effect when C=0; no separate B-to-router path is added.',
            'jobs': jobs}
    return {**body, 'protocol_sha256': canonical_hash(body)}
