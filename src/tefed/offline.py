"""Offline diagnostics from explicit review, condition, and executed-run records."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite, log
from numbers import Real
import os
from pathlib import Path
import tempfile

from .analysis import _run_protocol, analyze_run_dirs, summarize_seed_metric


PAPER_BIN_EDGES = (.4, .5, .6, .7, .8, .9, 1.)
FULL_BIN_EDGES = tuple(i / 10 for i in range(11))
CONDITION_SPECIALISTS = {'illumination': 1, 'occlusion': 2, 'context': 3,
                         'class_boundary': 4, 'sensing': 5}
CONDITION_ALIASES = {'illum': 'illumination', 'low_illumination': 'illumination',
                     'occ': 'occlusion', 'ctx': 'context', 'clutter': 'context',
                     'mat': 'class_boundary', 'maturity': 'class_boundary',
                     'maturity_boundary': 'class_boundary', 'sens': 'sensing'}
METHOD_SECTIONS = ('decision', 'loss', 'router', 'renderer', 'baseline', 'baselines',
                   'factorial', 'replay')


def _number(value, name, low=0., high=1.):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in [{low}, {high}]')
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _round(value):
    if type(value) is not int or value < 0:
        raise ValueError('round_index must be a nonnegative integer')
    return value


def event_key(row, join_on='candidate'):
    """Use explicit event, candidate, observation, or client-round identities."""
    if not isinstance(row, dict):
        raise ValueError('records must be JSON objects')
    if join_on == 'event_id':
        return (_text(row.get('event_id'), 'event_id'),)
    client = _text(row.get('client_id'), 'client_id')
    round_index = _round(row.get('round_index'))
    if join_on == 'client_round':
        return (client, round_index)
    identity = _text(row.get('local_identity'), 'local_identity')
    if join_on == 'identity':
        return (client, round_index, identity)
    if join_on != 'candidate':
        raise ValueError('join_on must be candidate, identity, client_round, or event_id')
    candidate = row.get('candidate_id', row.get('id'))
    if 'candidate_id' in row and 'id' in row and row['candidate_id'] != row['id']:
        raise ValueError('candidate_id and id disagree')
    return (client, round_index, identity, _text(candidate, 'candidate id'))


def _review_key(row):
    if not isinstance(row, dict):
        raise ValueError('review records must be JSON objects')
    if 'event_id' in row:
        return ('event', *event_key(row, 'event_id'))
    if 'local_identity' in row:
        return ('candidate', *event_key(row))
    return ('id', _text(row.get('id'), 'review id'),
            _text(row['client_id'], 'client_id') if 'client_id' in row else None,
            _round(row['round_index']) if 'round_index' in row else None)


def reliability_diagnostics(records, *, bin_edges=PAPER_BIN_EDGES, bin_policy='paper_s11_3'):
    """Apply the S11.3 binary review rule without inferring missing outcomes."""
    edges = [_number(value, 'bin edge') for value in bin_edges]
    if len(edges) < 2 or any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError('bin edges must be strictly increasing')
    bins = [{'lower': left, 'upper': right, 'upper_inclusive': i == len(edges) - 2,
             'count': 0, 'mean_reliability': None, 'accuracy': None}
            for i, (left, right) in enumerate(zip(edges, edges[1:]))]
    values = [[] for _ in bins]
    seen = set()
    for row in records:
        key = _review_key(row)
        if key in seen:
            raise ValueError(f'duplicate reviewed event identity: {key}')
        seen.add(key)
        reference = row.get('review_reference')
        if not isinstance(reference, dict) or reference.get('kind') != 'human_review':
            raise ValueError('review_reference must declare an independent human_review')
        _text(reference.get('id'), 'review_reference.id')
        if reference.get('independent', True) is not True or reference.get('renderer_reference', False) is not False:
            raise ValueError('renderer references cannot supply reviewed correctness')
        present = [name for name in ('reliability', 'R', 'r') if name in row]
        if not present:
            raise ValueError('missing reliability')
        reliability = _number(row[present[0]], 'reliability')
        if any(_number(row[name], name) != reliability for name in present[1:]):
            raise ValueError('reliability aliases disagree')
        outcome = None
        if 'reviewed_class_match' in row or 'reviewed_iou' in row:
            if type(row.get('reviewed_class_match')) is not bool:
                raise ValueError('reviewed_class_match must be an explicit boolean')
            iou = _number(row.get('reviewed_iou'), 'reviewed_iou')
            outcome = row['reviewed_class_match'] and iou >= .5
        if 'reviewed_correct' in row:
            if type(row['reviewed_correct']) is not bool:
                raise ValueError('reviewed_correct must be an explicit boolean')
            if outcome is not None and row['reviewed_correct'] != outcome:
                raise ValueError('reviewed_correct disagrees with class-match/IoU rule')
            outcome = row['reviewed_correct']
        if outcome is None:
            raise ValueError('missing independent reviewed correctness')
        if reliability < edges[0] or reliability > edges[-1]:
            raise ValueError('reliability lies outside bin edges; declare an expanded bin policy')
        index = min(bisect_right(edges, reliability) - 1, len(bins) - 1)
        values[index].append((reliability, int(outcome)))
    count = len(seen)
    ece = brier = nll = 0.
    for output, points in zip(bins, values):
        if not points:
            continue
        output['count'] = len(points)
        output['mean_reliability'] = sum(r for r, _ in points) / len(points)
        output['accuracy'] = sum(y for _, y in points) / len(points)
        ece += len(points) * abs(output['accuracy'] - output['mean_reliability'])
        brier += sum((r - y) ** 2 for r, y in points)
        for r, y in points:
            clipped = min(1 - 1e-6, max(1e-6, r))
            nll -= y * log(clipped) + (1 - y) * log(1 - clipped)
    return {'kind': 'reliability', 'count': count, 'status': 'completed' if count else 'empty',
            'bin_policy': bin_policy, 'bin_edges': edges, 'bins': bins,
            'ECE': ece / count if count else None, 'Brier': brier / count if count else None,
            'NLL': nll / count if count else None, 'nll_clip': [1e-6, 1 - 1e-6],
            'correctness_rule': 'assigned class matches independent human review and reviewed IoU >= 0.50; explicit reviewed_correct declares this same rule',
            'population': 'supplied reviewed events; no resampling or training runs inferred'}


def _route(row):
    route, weights = row.get('route'), row.get('weights')
    if not isinstance(route, list) or not isinstance(weights, list) or not route or len(route) != len(weights):
        raise ValueError('route and weights require matching nonempty lists')
    if any(type(head) is not int or not 0 <= head <= 5 for head in route) or len(set(route)) != len(route) or 0 not in route:
        raise ValueError('route must contain general 0 and unique expert IDs in 0..5')
    weights = [_number(value, 'route weight') for value in weights]
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError('route weights must sum to one')
    full = [0.] * 6
    for head, weight in zip(route, weights):
        full[head] = weight
    specialists = [head for head in route if head and full[head] > 0]
    top = min(specialists, key=lambda head: (-full[head], head)) if specialists else None
    return full, top


def routing_diagnostics(candidate_records, condition_records, *, join_on='candidate'):
    """Join external ordered condition labels; top-1 excludes general expert 0."""
    metadata = {}
    for row in condition_records:
        key = event_key(row, join_on)
        if key in metadata:
            raise ValueError(f'duplicate condition join identity: {key}')
        if row.get('condition_source') not in ('manifest', 'independent_annotation'):
            raise ValueError('condition_source must be manifest or independent_annotation')
        _text(row.get('reference_id'), 'condition reference_id')
        conditions = row.get('conditions')
        if not isinstance(conditions, list) or any(not isinstance(label, str) or not label.strip() for label in conditions):
            raise ValueError('conditions must be an explicitly ordered string list')
        metadata[key] = row
    groups, clients, missing, seen, used = {}, {}, [], set(), set()
    total = matched = eligible = no_specialist = 0
    for row in candidate_records:
        identity = ('event', *event_key(row, 'event_id')) if 'event_id' in row else ('candidate', *event_key(row))
        if identity in seen:
            raise ValueError(f'duplicate candidate event identity: {identity}')
        seen.add(identity)
        client = _text(row.get('client_id'), 'client_id')
        weights, top = _route(row)
        key = event_key(row, join_on)
        labels = metadata.get(key)
        first = labels['conditions'][0] if labels and labels['conditions'] else None
        if labels:
            used.add(key)
        if first is None:
            group = '__missing__'
            missing.append({'identity': list(identity), 'reason': 'missing_condition_record' if labels is None else 'empty_conditions'})
        else:
            group = CONDITION_ALIASES.get(first, first)
            if group not in (*CONDITION_SPECIALISTS, 'nominal'):
                raise ValueError(f'unknown first condition label: {first}')
        bucket = groups.setdefault(group, {'count': 0, 'weight_sums': [0.] * 6,
                                          'top1_specialist_counts': Counter(), 'correspondence_count': 0,
                                          'correspondence_matches': 0})
        client_bucket = clients.setdefault(client, {'count': 0, 'weight_sums': [0.] * 6,
                                                    'top1_specialist_counts': Counter()})
        for target in (bucket, client_bucket):
            target['count'] += 1
            target['weight_sums'] = [a + b for a, b in zip(target['weight_sums'], weights)]
            target['top1_specialist_counts'][str(top) if top is not None else 'none'] += 1
        total += 1
        no_specialist += top is None
        if group in CONDITION_SPECIALISTS:
            eligible += 1
            same = top == CONDITION_SPECIALISTS[group]
            matched += same
            bucket['correspondence_count'] += 1
            bucket['correspondence_matches'] += same
    for bucket in (*groups.values(), *clients.values()):
        bucket['mean_route_weights'] = [value / bucket['count'] for value in bucket.pop('weight_sums')]
        counts = bucket['top1_specialist_counts']
        bucket['top1_specialist_counts'] = {str(head): counts[str(head)] for head in range(1, 6)} | {'none': counts['none']}
    return {'kind': 'routing', 'count': total, 'join_on': join_on,
            'condition_rule': 'first externally recorded manifest/annotation condition; rule_flags are not labels',
            'specialist_order': CONDITION_SPECIALISTS, 'tie_break': 'lower specialist ID',
            'top1_rule': 'largest stored positive weight among specialists 1..5; general 0 excluded',
            'correspondence': {'count': eligible, 'matches': matched, 'fraction': matched / eligible if eligible else None,
                               'excluded_nominal': groups.get('nominal', {}).get('count', 0),
                               'excluded_missing': len(missing)},
            'without_active_specialist': no_specialist, 'groups': groups, 'clients': clients,
            'missing_conditions': missing, 'unused_condition_identities': [list(key) for key in metadata if key not in used],
            'interpretation': 'manifest-order correspondence; no specialist accuracy or unique condition truth inferred'}


def _paired_contract(path):
    config = json.loads((path / 'configuration.json').read_text(encoding='utf-8-sig'))
    environment = json.loads((path / 'environment.json').read_text(encoding='utf-8-sig'))
    status = json.loads((path / 'status.json').read_text(encoding='utf-8-sig'))
    rounds = config['experiment'].get('rounds')
    if (type(rounds) is not int or rounds <= 0 or status.get('status') != 'completed'
            or type(status.get('completed_rounds')) is not int or status['completed_rounds'] != rounds):
        raise ValueError(f'paired analysis requires a completed engine run: {path}')
    for name in ('final.pt', 'calibration.json'):
        if not (path / name).is_file():
            raise FileNotFoundError(f'missing completed run artifact: {path / name}')
    protocol = _run_protocol(config, environment, path)
    for key in ('manifest_sha256', 'calibration_sha256', 'initial_checkpoint_sha256'):
        if protocol['artifacts'][key] is None:
            raise ValueError(f'paired analysis requires recorded {key}')
    common = deepcopy(protocol)
    method_specific = {}
    for section in METHOD_SECTIONS:
        if section in common['configuration']:
            method_specific[section] = common['configuration'].pop(section)
    method_specific['method'] = common['configuration']['experiment'].pop('method')
    return common, method_specific


def paired_run_differences(left_dirs, right_dirs, *, expected_seeds=tuple(range(10)), metrics=('mAP50', 'AP75', 'mAP50_95')):
    """Subtract completed same-seed runs after within- and across-method checks."""
    expected_seeds = tuple(expected_seeds)
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or any(not isinstance(metric, str) or not metric for metric in metrics):
        raise ValueError('metrics must be distinct nonempty names')
    left_paths = [Path(path).resolve() for path in left_dirs]
    right_paths = [Path(path).resolve() for path in right_dirs]
    if set(left_paths) & set(right_paths):
        raise ValueError('left and right runs must be distinct directories')
    left = analyze_run_dirs(left_paths, expected_seeds=expected_seeds, metrics=metrics)
    right = analyze_run_dirs(right_paths, expected_seeds=expected_seeds, metrics=metrics)
    for side in (left, right):
        if len({row['method'] for row in side['runs']}) != 1:
            raise ValueError('each paired side requires exactly one method')
    contracts = {}
    for side, paths, analysis in (('left', left_paths, left), ('right', right_paths, right)):
        for path, row in zip(paths, analysis['runs']):
            common, method_specific = _paired_contract(path)
            contracts[(side, row['seed'])] = (common, method_specific)
            row['files_sha256'].update({name: sha256((path / name).read_bytes()).hexdigest()
                                       for name in ('status.json', 'final.pt', 'calibration.json')
                                       if (path / name).is_file()})
    for seed in map(str, expected_seeds):
        if contracts[('left', seed)][0] != contracts[('right', seed)][0]:
            raise ValueError(f'paired seed {seed} has different data, calibration, initialization, evaluation, training, model, or execution settings')
    left_summary = {(row['view'], row['metric']): row for row in left['summary']}
    right_summary = {(row['view'], row['metric']): row for row in right['summary']}
    if set(left_summary) != set(right_summary):
        raise ValueError('paired runs must report identical evaluation views and metrics')
    result = []
    for key in sorted(left_summary):
        a, b = left_summary[key], right_summary[key]
        per_seed = [{'seed': seed, 'left': x, 'right': y, 'right_minus_left': y - x}
                    for seed, x, y in zip(a['seed_ids'], a['values'], b['values'])]
        result.append({'view': key[0], 'metric': key[1], 'per_seed': per_seed,
                       **summarize_seed_metric(per_seed, 'right_minus_left', expected_seeds)})
    first_seed = str(expected_seeds[0])
    common = contracts[('left', first_seed)][0]
    encoded = json.dumps(common, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return {'kind': 'paired', 'direction': 'right_minus_left', 'summary': result,
            'left': left, 'right': right,
            'shared_contract_sha256': sha256(encoded).hexdigest(), 'shared_artifacts': common['artifacts'],
            'comparability': 'same recorded data/calibration/initialization, all model, evaluation, optimizer, experiment and other shared settings; fixed complete protocol within each method',
            'allowed_method_sections': list(METHOD_SECTIONS),
            'method_specific_settings': {side: contracts[(side, first_seed)][1] for side in ('left', 'right')},
            'inference': 'arithmetic same-seed differences and sample SD only; no confidence interval, p-value, or independent sampling inferred'}


def input_records(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSONL at {path}:{number}') from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('JSONL records must be objects')
    return rows, {'path': str(path), 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}


def write_new_json(path, result):
    """Commit a new result atomically; never replace an existing path."""
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path.resolve()
