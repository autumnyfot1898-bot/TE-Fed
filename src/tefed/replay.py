"""Immutable case pools and strict recorded-exposure replay."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from . import core
from .experiments import artifact_bindings, canonical_hash, file_hash


class ExposureError(ValueError):
    pass


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ExposureError(f'{name} must be a nonnegative integer')
    return value


def _training_binding(config):
    resolved = deepcopy(config)
    recorded_policies = resolved.pop('resolved_baseline_policies', None)
    if recorded_policies is not None:
        from .baselines import policy_schedule
        if recorded_policies != policy_schedule(resolved['experiment'].get('method', 'tefed'), resolved):
            raise ValueError('Recorded baseline policies differ from the training configuration')
    resolved.pop('_config_path', None); resolved.pop('replay', None)
    for key in ('seed', 'seeds', 'output', 'manifest', 'calibration'):
        resolved['experiment'].pop(key, None)
    resolved.get('model', {}).pop('initial_checkpoint', None)
    resolved.get('decision', {}).pop('selection_mode', None)
    return canonical_hash(resolved)


def freeze_pool(cases, source_states, directory, *, provenance=None):
    """Copy private assets into a new content-bound pool directory."""
    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(directory)
    cases = deepcopy(list(cases)); states = deepcopy(source_states)
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Frozen pool contains duplicate case IDs')
    sources = []
    for case in cases:
        _integer(case['round'], 'case creation round')
        descriptor = core.validate_descriptor(case['d'], states)
        if descriptor['round'] != case['round']:
            raise ValueError('Case and source descriptor creation rounds differ')
        if not math.isfinite(case['fidelity']) or not 0 <= case['fidelity'] <= 1:
            raise ValueError('Case fidelity must be in [0,1]')
        if not case.get('targets') or not case.get('rgb'):
            raise ValueError('Frozen cases require renderer annotations and RGB assets')
        for key in ('rgb', 'depth', 'nir', 'instance_mask'):
            if case.get(key):
                path = Path(case[key]).expanduser().resolve(strict=True)
                sources.append((case, key, path, file_hash(path)))
    directory.mkdir(parents=True)
    assets = {}
    (directory / 'assets').mkdir()
    for case, key, path, digest in sources:
        relative = f'assets/{digest}{path.suffix.lower()}'
        target = directory / relative
        if not target.exists():
            shutil.copyfile(path, target)
        assets[relative] = {'sha256': digest, 'bytes': target.stat().st_size}
        case[key] = relative
    body = {'schema_version': 1, 'cases': cases, 'source_states': states,
            'assets': assets, 'provenance': provenance or {}}
    manifest = directory / 'pool.json'
    _write(manifest, {**body, 'pool_sha256': canonical_hash(body)})
    return manifest


class FrozenPool:
    def __init__(self, manifest, expected_hash=None):
        self.path = Path(manifest).expanduser().resolve(strict=True)
        raw = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.sha256 = raw.pop('pool_sha256', None)
        if raw.get('schema_version') != 1 or self.sha256 != canonical_hash(raw):
            raise ValueError('Frozen pool manifest integrity check failed')
        if expected_hash is not None and expected_hash != self.sha256:
            raise ValueError('Frozen pool differs from its expected hash')
        self._body = raw
        self._manifest_hash = file_hash(self.path)
        if len({case['id'] for case in raw['cases']}) != len(raw['cases']):
            raise ValueError('Frozen pool contains duplicate case IDs')
        for case in raw['cases']:
            _integer(case['round'], 'case creation round')
            descriptor = core.validate_descriptor(case['d'], raw['source_states'])
            if descriptor['round'] != case['round']:
                raise ValueError('Case and source descriptor creation rounds differ')
            for key in ('rgb', 'depth', 'nir', 'instance_mask'):
                if case.get(key) and case[key] not in raw['assets']:
                    raise ValueError('Frozen case refers to an unbound asset')
        self.verify()

    def verify(self):
        if file_hash(self.path) != self._manifest_hash or canonical_hash(self._body) != self.sha256:
            raise ValueError('Frozen pool changed during replay')
        for relative, binding in self._body['assets'].items():
            path = (self.path.parent / relative).resolve(strict=True)
            if not path.is_relative_to(self.path.parent):
                raise ValueError('Frozen assets must stay inside the pool bundle')
            if path.stat().st_size != binding['bytes'] or file_hash(path) != binding['sha256']:
                raise ValueError(f'Frozen asset changed: {relative}')

    @property
    def cases(self):
        cases = deepcopy(self._body['cases'])
        for case in cases:
            for key in ('rgb', 'depth', 'nir', 'instance_mask'):
                if case.get(key):
                    case[key] = str((self.path.parent / case[key]).resolve())
        return cases

    @property
    def source_states(self):
        return deepcopy(self._body['source_states'])


def _actual(ledger):
    actual = {'correction_presentations': _integer(ledger.get('presentations', {}).get('correction'), 'correction presentations'),
            'attempted_optimizer_steps': _integer(ledger.get('attempted_optimizer_steps'), 'attempted updates'),
            'applied_optimizer_steps': _integer(ledger.get('optimizer_steps'), 'applied updates')}
    if actual['applied_optimizer_steps'] > actual['attempted_optimizer_steps']:
        raise ExposureError('Applied updates exceed attempts')
    if 'amp_skipped_steps' in ledger and actual['attempted_optimizer_steps'] - _integer(ledger['amp_skipped_steps'], 'AMP skipped steps') != actual['applied_optimizer_steps']:
        raise ExposureError('Attempted/applied/AMP-skipped counts disagree')
    return actual


def build_exposure_protocol(pool_manifest, reference_runs, output):
    pool = FrozenPool(pool_manifest)
    cases = {case['id']: case for case in pool.cases}
    units = []; bindings = None; training_binding = None; seeds = set(); clients = None; rounds = None
    for source in reference_runs:
        root = Path(source)
        environment = json.loads((root / 'environment.json').read_text(encoding='utf-8-sig'))
        config = json.loads((root / 'configuration.json').read_text(encoding='utf-8-sig'))
        status = json.loads((root / 'status.json').read_text(encoding='utf-8-sig'))
        if status.get('status') != 'completed':
            raise ValueError('Exposure reference runs must be completed')
        seed = _integer(environment['seed'], 'seed')
        if seed in seeds:
            raise ValueError('Duplicate reference seed')
        seeds.add(seed)
        current = {key: environment.get(key) for key in ('manifest_sha256', 'calibration_sha256', 'initial_checkpoint_sha256')}
        if bindings is not None and current != bindings:
            raise ValueError('Exposure references require common data, calibration and initialization')
        bindings = current
        current_training = _training_binding(config)
        if training_binding not in (None, current_training):
            raise ValueError('Exposure references require identical training configurations')
        training_binding = current_training
        ledgers = [json.loads(line) for line in (root / 'rounds.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        count = config['experiment'].get('rounds', 30)
        if [row['round_index'] for row in ledgers] != list(range(count)) or rounds not in (None, count):
            raise ValueError('Exposure references need identical complete round sequences')
        rounds = count
        for row in ledgers:
            current_clients = sorted(row['clients'])
            if clients not in (None, current_clients):
                raise ValueError('Exposure references require identical participating clients')
            clients = current_clients
            for client in clients:
                selected = row['reuse_selected_ids'][client]
                if len(set(selected)) != len(selected):
                    raise ValueError('Reference selections must contain unique cases')
                if any(case not in cases or cases[case]['round'] >= row['round_index'] for case in selected):
                    raise ValueError('Reference selections must be eligible members of the frozen pool')
                actual = _actual(row['clients'][client])
                expected = {'case_count': len(selected), **actual}
                if actual['correction_presentations'] != len(selected) * config['experiment'].get('local_epochs', 2):
                    raise ValueError('Reference presentations disagree with case count and local epochs')
                if actual['applied_optimizer_steps'] > actual['attempted_optimizer_steps']:
                    raise ValueError('Applied updates cannot exceed attempts')
                units.append({'seed': seed, 'client_id': client, 'round_index': row['round_index'], 'expected': expected})
    if not units:
        raise ValueError('Exposure protocol needs at least one completed reference run')
    body = {'schema_version': 1, 'pool_sha256': pool.sha256, 'bindings': bindings,
            'training_configuration_sha256': training_binding,
            'clients': clients, 'rounds': rounds, 'seed_ids': sorted(seeds), 'units': units,
            'semantics': 'Strict actual counts; AMP skips may invalidate an arm. No below-gate filling or repeated cases.'}
    target = Path(output).expanduser().resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    _write(target, {**body, 'protocol_sha256': canonical_hash(body)})
    return target


def validate_replay_before(items, config, control):
    expected = control['expected']
    selected = [str(item['case']['id']) for item in items if item['kind'] == 'correction']
    if len(selected) != len(set(selected)) or sorted(selected) != sorted(control['selected_case_ids']):
        raise ExposureError('Replay training cases differ from the selected frozen cases')
    epochs = _integer(config['experiment'].get('local_epochs', 2), 'local epochs')
    batch = _integer(config['experiment'].get('batch_size', 4), 'batch size')
    if not batch or not epochs:
        raise ExposureError('Replay batch size and epochs must be positive')
    actual = {'case_count': len(selected), 'correction_presentations': len(selected) * epochs,
              'attempted_optimizer_steps': math.ceil(len(items) / batch) * epochs}
    differences = {key: {'expected': expected[key], 'actual': value} for key, value in actual.items() if expected[key] != value}
    if differences:
        raise ExposureError(f'Replay pre-training exposure mismatch: {differences}')


def validate_replay_after(ledger, control):
    actual = _actual(ledger)
    differences = {key: {'expected': control['expected'][key], 'actual': value}
                   for key, value in actual.items() if control['expected'][key] != value}
    if differences:
        raise ExposureError(f'Replay actual exposure mismatch (including possible AMP skips): {differences}')
    return {'pool_sha256': control['pool_sha256'], 'protocol_sha256': control['protocol_sha256'],
            'seed': control['seed'], 'client_id': control['client_id'], 'round_index': control['round_index'],
            'expected': deepcopy(control['expected']), 'actual': actual, 'matched': True}


class ReplayController:
    """Pure per-unit validation; resume accounting is reconstructed from ledgers."""
    def __init__(self, config, clients, rounds, seed):
        settings = config['replay']
        self.arm = settings['arm']
        if self.arm not in ('matched', 'independent'):
            raise ValueError('Replay arm must be matched or independent')
        self.frozen = FrozenPool(settings['pool_manifest'], settings.get('expected_pool_sha256'))
        self._protocol_path = Path(settings['exposure_protocol']).expanduser().resolve(strict=True)
        self._protocol_file_hash = file_hash(self._protocol_path)
        self.protocol = json.loads(self._protocol_path.read_text(encoding='utf-8-sig'))
        self.protocol_sha256 = self.protocol.pop('protocol_sha256', None)
        if self.protocol.get('schema_version') != 1 or canonical_hash(self.protocol) != self.protocol_sha256:
            raise ValueError('Exposure protocol integrity check failed')
        if settings.get('expected_protocol_sha256', self.protocol_sha256) != self.protocol_sha256:
            raise ValueError('Exposure protocol differs from its expected hash')
        if self.protocol['pool_sha256'] != self.frozen.sha256 or sorted(map(str, clients)) != self.protocol['clients'] or rounds != self.protocol['rounds']:
            raise ValueError('Replay pool, participating clients or rounds differ from the protocol')
        if artifact_bindings(config) != self.protocol['bindings']:
            raise ValueError('Replay data, calibration or initial checkpoint differs from the protocol')
        if _training_binding(config) != self.protocol['training_configuration_sha256']:
            raise ValueError('Replay training configuration differs from the exposure reference')
        if seed not in self.protocol['seed_ids']:
            raise ValueError('Replay seed is not declared by the exposure protocol')
        self.seed = seed
        self.units = {(u['seed'], str(u['client_id']), u['round_index']): u['expected'] for u in self.protocol['units']}
        if len(self.units) != len(self.protocol['units']):
            raise ValueError('Duplicate replay exposure unit')
        for client in clients:
            for round_index in range(rounds):
                expected = self.units.get((seed, str(client), round_index))
                if expected is None or set(expected) != {'case_count', 'correction_presentations', 'attempted_optimizer_steps', 'applied_optimizer_steps'}:
                    raise ValueError('Exposure protocol lacks a complete client-round unit')
                for key, value in expected.items():
                    _integer(value, key)
                if expected['applied_optimizer_steps'] > expected['attempted_optimizer_steps']:
                    raise ValueError('Exposure protocol has more applied updates than attempts')
        self._protocol_hash = canonical_hash(self.protocol)

    @property
    def pool(self):
        return self.frozen.cases

    @property
    def source_states(self):
        return self.frozen.source_states

    def metadata(self):
        return {'schema_version': 1, 'pool_sha256': self.frozen.sha256,
                'protocol_sha256': self.protocol_sha256, 'arm': self.arm, 'seed': self.seed}

    def state_dict(self):
        return self.metadata()

    def load_state_dict(self, state):
        if state != self.metadata():
            raise ValueError('Resume replay state differs from the bound protocol')

    def verify(self):
        self.frozen.verify()
        if canonical_hash(self.protocol) != self._protocol_hash or file_hash(self._protocol_path) != self._protocol_file_hash:
            raise ValueError('Exposure protocol changed during replay')

    def _control(self, client, round_index, selected):
        return {**self.metadata(), 'client_id': str(client), 'round_index': round_index,
                'selected_case_ids': [str(case['id']) for case in selected],
                'expected': deepcopy(self.units[(self.seed, str(client), round_index)])}

    def select(self, state, client, round_index, decision=None):
        quota = self.units[(self.seed, str(client), round_index)]['case_count']
        decision = {**(decision or {}), 'reuse_cap': quota}
        if self.arm == 'matched':
            result = core.select_reuse(self.pool, state, round_index, decision)
        else:
            ranked = sorted((case for case in self.pool if case['round'] < round_index),
                            key=lambda case: (-case['fidelity'], -case['d'].get('priority', 0), str(case['id'])))
            selected, patterns, sources = [], {}, {}
            for case in ranked:
                pattern, source = case['d']['pattern'], case['d']['source']
                if len(selected) >= quota:
                    break
                if patterns.get(pattern, 0) >= decision.get('pattern_cap', 2) or sources.get(source, 0) >= decision.get('source_cap', 1):
                    continue
                selected.append(case); patterns[pattern] = patterns.get(pattern, 0) + 1; sources[source] = sources.get(source, 0) + 1
            result = {'selected': selected, 'audit': [{'case_id': case['id'], 'reason': 'state_independent_selected'} for case in selected]}
        if len(result['selected']) != quota:
            raise ExposureError(f'Replay {self.arm} cannot supply {quota} eligible diverse cases for {client}, round {round_index}; no filling is allowed')
        return {**result, 'control': self._control(client, round_index, result['selected'])}

    def before_train(self, items, client, round_index, config, control=None):
        selected = [item['case'] for item in items if item['kind'] == 'correction']
        validate_replay_before(items, config, control or self._control(client, round_index, selected))

    def after_train(self, ledger, client, round_index, control=None):
        return validate_replay_after(ledger, control or self._control(client, round_index, []))

    def verify_history(self, history):
        cases = {case['id']: case for case in self.pool}
        if [row['round_index'] for row in history] != list(range(len(history))):
            raise ExposureError('Resumed replay history is not a complete round prefix')
        for row in history:
            if sorted(row['clients']) != self.protocol['clients']:
                raise ExposureError('Resumed replay history has different participating clients')
            for client, ledger in row['clients'].items():
                selected = row['reuse_selected_ids'][client]
                expected = self.units[(self.seed, str(client), row['round_index'])]
                if len(set(selected)) != expected['case_count'] or len(selected) != expected['case_count']:
                    raise ExposureError('Resumed selection count differs from exposure protocol')
                if any(case not in cases or cases[case]['round'] >= row['round_index'] for case in selected):
                    raise ExposureError('Resumed selections are not eligible frozen cases')
                record = ledger.get('replay')
                if record is None or record.get('pool_sha256') != self.frozen.sha256 or record.get('protocol_sha256') != self.protocol_sha256:
                    raise ExposureError('Resumed ledger lacks the bound replay identities')
                actual_record = self.after_train(ledger, client, row['round_index'])
                if record != actual_record:
                    raise ExposureError('Resumed replay accounting record disagrees with its actual ledger')
