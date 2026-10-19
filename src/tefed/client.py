"""Client tasks shared by local execution and remote workers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
import random

import numpy as np
import torch


def module_state(module):
    return {key: value.detach().cpu().numpy().copy() for key, value in module.state_dict().items()}


def load_module_state(module, values):
    current = module.state_dict()
    if set(current) != set(values):
        raise ValueError('Module state keys differ')
    restored = {}
    for key, target in current.items():
        value = np.asarray(values[key])
        if value.shape != tuple(target.shape) or value.dtype != target.detach().cpu().numpy().dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid module state: {key}')
        restored[key] = torch.as_tensor(value.copy(), device=target.device, dtype=target.dtype)
    module.load_state_dict(restored, strict=True)


def capture_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().numpy().copy(),
            'cuda': [value.cpu().numpy().copy() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(torch.as_tensor(state['torch'].copy(), dtype=torch.uint8))
    if state.get('cuda'):
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count differs')
        torch.cuda.set_rng_state_all([torch.as_tensor(value.copy(), dtype=torch.uint8) for value in state['cuda']])


def initialize_client_state(model, router, manifest, calibration, config, *, client_id, prior=None):
    from .engine import prepared_sample, assess_frame, _state_statistics
    from .baselines import policy_dict
    from .state import initialize_state
    records = manifest.by_split('validation', client_id)
    if not records:
        raise ValueError(f'Client {client_id} has no validation pool')
    prior = prior or calibration.get('state_prior', {'q': .5, 'p': .5, 'h': [.2] * 5})
    samples = [prepared_sample(record, config) for record in records]
    policy = policy_dict(config['experiment'].get('method', 'tefed'), 0, 0, config)
    assessed = [assess_frame(model, router, sample, prior, calibration, config, experts=policy['use_experts'], policy=policy) for sample in samples]
    return initialize_state(_state_statistics(samples, assessed, calibration))


def execute_client_round(model, router, manifest, state, calibration, config, *, client_id,
                         round_index, selected_cases=(), run_secret, persistent_state=None, replay_control=None):
    from . import core
    from .engine import (prepared_sample, assess_frame, _state_statistics, _sample_case,
                         train_local, tree_delta, tree_add, validation_loss)
    from .model import sample_tensor
    from .state import update_state, state_specialist_probabilities
    from .baselines import policy_dict, make_ema_teacher, batchnorm_state_keys
    from .diagnostics import renderer_target_observation

    client_id = str(client_id)
    method = config['experiment'].get('method', 'tefed')
    policy = policy_dict(method, round_index, 0, config)
    persistent = deepcopy(persistent_state or {})
    if set(persistent) - {'local_model', 'teacher', 'bn', 'new_client_model'}:
        raise ValueError('Unknown persistent client state')
    base = model.export_state()
    device = next(model.parameters()).device
    selected_events = []
    if any(int(case['round']) >= round_index for case in selected_cases):
        raise ValueError('Same-round correction reuse is forbidden')
    for case in selected_cases:
        sample = _sample_case(case, config)
        heads = tuple(case['d']['route'])
        predictions = model.predict_heads(sample_tensor(sample, device, rgb_only=config['model'].get('detector_rgb_only', False)), heads,
            confidence=float(config.get('evaluation', {}).get('prediction_confidence', .01)),
            max_detections=int(config.get('evaluation', {}).get('max_detections', 100)), temperature=float(calibration['T_C']))
        observation = renderer_target_observation({str(m): predictions[m][0] for m in heads}, heads, case['d']['weights'],
            sample['targets'][int(case.get('focal_target_index', 0))], scope='current round-start bank before local training')
        selection={'client_id':client_id,'round_index':round_index,'case_id':str(case['id'])}
        selection_id=hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        descriptor=case['d']
        selected_events.append({**selection,'selection_event_id':selection_id,
            'descriptor_id':descriptor['id'],'source_observation_key':descriptor['source'],
            'pattern_key':descriptor['pattern'],'creation_round':int(case['round']),
            'source_route':list(descriptor['route']),'source_weights':list(descriptor['weights']),**observation})
    labeled = [prepared_sample(record, config) for record in manifest.by_split('labeled', client_id)]
    if not labeled:
        raise ValueError(f'Client {client_id} has no labeled observations')
    unlabeled = [prepared_sample(record, config) for record in manifest.by_split('unlabeled', client_id)] if policy['use_unlabeled'] else []
    old_teacher = None
    if persistent.get('teacher') is not None:
        old_teacher = make_ema_teacher(model)
        load_module_state(old_teacher, persistent['teacher'])
    assessment_model = old_teacher if old_teacher is not None and policy['use_teacher'] and not policy['reset_teacher_each_round'] else model
    assessed = [assess_frame(assessment_model, router, sample, state, calibration, config,
                            round_id=round_index, experts=policy['use_experts'], policy=policy) for sample in unlabeled]
    active = {0}
    for assessment in assessed:
        for candidate in assessment['candidates']:
            active.update(candidate['route'])
    if policy['use_experts'] and active == {0}:
        demand = state_specialist_probabilities(state)
        active.update(int(key) for key in sorted(demand, key=lambda key: (-demand[key], int(key)))[:2])
    supervised = {head: 1 / len(active) for head in active}
    items = [{'kind': 'labeled', 'sample': sample, 'route_weights': supervised} for sample in labeled]
    descriptors, source_states, candidate_audit = [], {}, []
    secret = run_secret.encode('utf-8') if isinstance(run_secret, str) else run_secret
    if not isinstance(secret, bytes) or not secret:
        raise ValueError('Client tasks require a source identifier key')
    for sample, assessment in zip(unlabeled, assessed):
        qualified = [candidate for candidate in assessment['candidates'] if candidate['disposition'] == 'qualified']
        if qualified:
            pseudo = {**sample, 'targets': [{'box': candidate['box'], 'class_id': candidate['class_id']} for candidate in qualified]}
            items.append({'kind': 'pseudo', 'sample': pseudo, 'candidates': qualified})
        identity = hmac.new(secret, str(sample['id']).encode('utf-8'), 'sha256').hexdigest()[:32]
        for candidate in assessment['candidates']:
            candidate_audit.append({'client_id':client_id,'round_index':round_index,'local_identity':identity,
                **candidate,'candidate_id':str(candidate.get('id',candidate.get('anchor_id','')))})
            if policy['use_corrections'] and candidate['disposition'] == 'correction':
                descriptor = core.make_descriptor(candidate, candidate['attrs'], candidate['availability'], candidate['rule_flags'],
                                                  client_id, sample['id'], round_index, state, secret)
                source_states[descriptor['source_state_key']] = deepcopy(state)
                descriptors.append(descriptor)
                candidate_audit[-1].update(descriptor_id=descriptor['id'],source_observation_key=descriptor['source'])
    items.extend({'kind': 'correction', 'sample': _sample_case(case, config), 'case': case} for case in selected_cases)
    student = deepcopy(model)
    if method == 'local' and persistent.get('local_model') is not None:
        load_module_state(student, persistent['local_model'])
    if method == 'fedbn' and persistent.get('bn') is not None:
        values = module_state(student)
        values.update(persistent['bn'])
        load_module_state(student, values)
    teacher = None
    if policy['use_teacher']:
        teacher = make_ema_teacher(student) if policy['reset_teacher_each_round'] or old_teacher is None else old_teacher
    if replay_control is not None:
        from .replay import validate_replay_before
        validate_replay_before(items, config, replay_control)
    ledger = train_local(student, items, config, round_id=round_index, method=method, teacher=teacher, reference=model)
    ledger['assessment_policy']=deepcopy(policy)
    dispositions=[candidate['disposition'] for assessment in assessed for candidate in assessment['candidates']]
    ledger['candidate_counts']={name:dispositions.count(name) for name in ('qualified','correction','suppressed')}
    ledger['candidate_counts']['total']=len(dispositions)
    for event in selected_events:
        tracked=ledger['case_presentations'].get(event['case_id'],{})
        event.update(actual_presentations=tracked.get('presentations',0),
            supervised_presentations=tracked.get('supervised_presentations',0),
            detection_supervision_enabled=tracked.get('detection_supervision_enabled',False),
            attempted_step_ids=tracked.get('attempted_step_ids',[]),
            applied_step_ids=tracked.get('applied_step_ids',[]),
            amp_skipped_step_ids=tracked.get('amp_skipped_step_ids',[]))
    if replay_control is not None:
        from .replay import validate_replay_after
        ledger['replay'] = validate_replay_after(ledger, replay_control)
    next_persistent = {'local_model': module_state(student) if method == 'local' else None,
                       'teacher': module_state(teacher) if teacher is not None else None, 'bn': None}
    if method == 'fedbn':
        keys = batchnorm_state_keys(student)
        next_persistent['bn'] = {key: value for key, value in module_state(student).items() if key in keys}
    after = student.export_state()
    delta = tree_delta(after, base)
    upload = None
    if policy['use_experts']:
        candidates = [candidate for assessment in assessed for candidate in assessment['candidates'] if candidate.get('valid') and candidate.get('R') is not None]
        rbar = [float(np.mean([candidate['R'] for candidate in candidates if head in candidate['route']]))
                if any(head in candidate['route'] for candidate in candidates) else 0. for head in range(6)]
        gain = []
        validation_records = manifest.by_split('validation', client_id)
        probe = deepcopy(model)
        for head in range(6):
            before_loss = validation_loss(model, validation_records, config, head)
            probe.import_state(tree_add(base, delta, head=head))
            after_loss = validation_loss(probe, validation_records, config, head)
            gain.append(float(np.clip((before_loss - after_loss) / (before_loss + 1e-8), 0, 1)))
        upload = {'client_id': client_id, 'delta': delta, 'bar_R': rbar, 'gain': gain}
        ledger.update(bar_R=rbar, validation_gain=gain)
    next_state = update_state(state, _state_statistics(unlabeled, assessed, calibration)) if unlabeled else deepcopy(state)
    return {'after': after, 'delta': delta, 'upload': upload, 'sample_count': len(labeled), 'next_state': next_state,
            'descriptors': descriptors, 'source_states': source_states, 'candidate_audit': candidate_audit,
            'selected_events': selected_events, 'ledger': ledger, 'persistent_state': next_persistent}


@torch.no_grad()
def evaluate_client(model, router, manifest, state, calibration, config, *, client_id, persistent_state=None):
    from .engine import prepared_sample, assess_frame
    from .data import apply_unseen
    from .evaluation_manifest import evaluation_record
    method = config['experiment'].get('method', 'tefed')
    experts = method in {'server_moe', 'generic_hc', 'tefed'}
    persistent = persistent_state or {}
    before, router_before = module_state(model), module_state(router)
    prior = calibration.get('state_prior', {'q': .5, 'p': .5, 'h': [.2] * 5})
    result = {}
    try:
        for view in config.get('evaluation', {}).get('views', ['participating', 'unseen', 'new-client', 'hard']):
            if view not in {'participating', 'unseen', 'new-client', 'hard'}:
                raise ValueError(f'Unknown evaluation view: {view}')
            records = manifest.by_split('hard' if view == 'hard' else 'test', client_id)
            normalized = {str(key).lower().replace('_', '-'): value for key, value in manifest.metadata.get('views', {}).items()}
            if view in normalized:
                allowed = set(map(str, normalized[view]))
                records = tuple(record for record in manifest.records if str(record.client_id) == str(client_id) and record.id in allowed)
            load_module_state(model, before)
            if method == 'local':
                key = 'new_client_model' if view == 'new-client' else 'local_model'
                if persistent.get(key) is None:
                    raise ValueError(f'Local-only evaluation requires {key}')
                load_module_state(model, persistent[key])
            elif method == 'fedbn' and view != 'new-client':
                if persistent.get('bn') is None:
                    raise ValueError('FedBN evaluation requires the client BN state')
                values = dict(before)
                values.update(persistent['bn'])
                load_module_state(model, values)
            targets, predictions, audit_records = {}, {}, []
            for record in records:
                sample = prepared_sample(record, config)
                if view == 'unseen':
                    sample = apply_unseen(sample, identity=record.id,
                        perturbation=config.get('evaluation', {}).get('unseen', {'rgb_brightness': config.get('evaluation', {}).get('unseen_brightness', .65)}),
                        seed=int(config['experiment'].get('seed', 0)))
                frozen = prior if view == 'new-client' else state
                assessed = assess_frame(model, router, sample, frozen, calibration, config, experts=experts)
                targets[record.id] = sample['targets']
                predictions[record.id] = [{'box': candidate['box'], 'class_id': candidate['class_id'], 'score': float(candidate['score'])}
                                          for candidate in assessed['candidates']]
                audit_records.append(evaluation_record(record,sample,config,view,
                    state_source='calibration_prior' if view=='new-client' else 'frozen_client_state'))
            result[view] = {'targets': targets, 'predictions': predictions,'audit_records':audit_records}
    finally:
        load_module_state(model, before)
        load_module_state(router, router_before)
    return {'views': result}


def execute_client_task(config, manifest, model, router, calibration, *, action, client_id,
                        round_index=0, state=None, selected_cases=(), persistent_state=None, run_secret=None, replay_control=None):
    if action == 'initialize':
        return {'state': initialize_client_state(model, router, manifest, calibration, config, client_id=client_id, prior=state)}
    if action == 'client_round':
        return execute_client_round(model, router, manifest, state, calibration, config, client_id=client_id,
            round_index=round_index, selected_cases=selected_cases, run_secret=run_secret, persistent_state=persistent_state, replay_control=replay_control)
    if action == 'evaluate':
        return evaluate_client(model, router, manifest, state, calibration, config, client_id=client_id, persistent_state=persistent_state)
    raise ValueError(f'Unknown client action: {action}')
