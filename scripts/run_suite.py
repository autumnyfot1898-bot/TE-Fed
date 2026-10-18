"""Plan or run comparisons and ablations."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256


def plans(name):
    from tefed.baselines import METHOD_REGISTRY
    if name=='factorial':
        from tefed.experiments import factorial_plans
        return factorial_plans()
    if name=='replay':
        return [(arm,{'experiment':{'method':'tefed'},'replay':{'arm':arm}}) for arm in ('matched','independent')]
    if name=='comparisons':
        return [(method,{'experiment':{'method':method}}) for method in METHOD_REGISTRY]
    if name=='mechanisms':
        return [('source_stored',{'decision':{'verification_route':'stored'}}),
                ('source_current',{'decision':{'verification_route':'current'}}),
                ('source_cyclic',{'decision':{'verification_route':'cyclic'}}),
                ('reuse_matched',{'decision':{'selection_mode':'matched'}}),
                ('reuse_independent',{'decision':{'selection_mode':'independent'}})]
    if name=='budgets':
        return [(f'render_{b}_reuse_{k}',{'decision':{'render_budget':b,'reuse_cap':k}})
                for b,k in [(576,6),(1152,3),(1152,6),(1152,9),(2304,6)]]
    if name=='components':
        return [('full',{}),('no_alignment',{'loss':{'rho':0.},'decision':{'omit_cross_modal':True}}),
                ('rgb_only',{'model':{'disable_auxiliary':True},'decision':{'omit_cross_modal':True}}),
                ('correction_loss_zero',{'loss':{'beta':0.}}),
                ('correction_path_off',{'experiment':{'method':'server_moe'}})]
    raise ValueError(name)


def merge(base,patch):
    result=deepcopy(base)
    for key,value in patch.items():
        result[key]=merge(result.get(key,{}),value) if isinstance(value,dict) else value
    return result


def _atomic_text(path, content):
    path = Path(path)
    if path.is_file() and path.read_text(encoding='utf-8') == content:
        return
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write(path, value):
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _job_config(config, manifest, job):
    from tefed.baselines import policy_dict
    resolved = merge(config, job['patch'])
    if manifest.get('version') == 'tefed-factorial-v1':
        resolved['factorial']['plan_sha256'] = manifest['protocol_sha256']
    resolved['experiment']['seed'] = job['seed']
    method = resolved['experiment'].get('method', 'tefed')
    resolved['experiment']['method'] = method
    if not policy_dict(method, 0, 0)['use_experts']:
        resolved['model']['specialists'] = 0
    return resolved


def _binding_scope(config):
    from tefed.experiments import canonical_hash
    renderer = config.get('renderer', {})
    active = config['experiment']['method'] in {'tefed', 'generic_hc'} \
        and renderer.get('enabled', True) and config.get('decision', {}).get('render_budget', 1152) > 0 \
        and not config.get('replay')
    paths = {}
    for section, key in (('experiment', 'manifest'), ('experiment', 'calibration'),
                         ('model', 'initial_checkpoint'), ('replay', 'pool_manifest'),
                         ('replay', 'exposure_protocol')):
        value = config.get(section, {}).get(key)
        paths[f'{section}.{key}'] = str(Path(value).expanduser().resolve()) if value else None
    return canonical_hash({'paths': paths, 'renderer': renderer if active else None,
                           'local': config.get('communication', {}).get('mode', 'local') == 'local'})


def make_manifest(config, name, root):
    from tefed.data import load_manifest
    from tefed.engine import source_secret, validate_run_config
    from tefed.experiments import canonical_hash, factorial_manifest, validate_factorial_config
    from tefed.resume import file_hash, run_binding
    seeds = config['experiment'].get('seeds', list(range(10)))
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Seed IDs must be distinct nonnegative integers')
    if name == 'factorial':
        manifest = factorial_manifest(config, seeds, root)
    else:
        jobs = [{'variant': variant, 'seed': seed, 'patch': patch,
                 'output': str(root / variant / f'seed_{seed}')}
                for variant, patch in plans(name) for seed in seeds]
        manifest = {'implementation': 'TE-Fed', 'plan': name, 'seed_ids': seeds, 'jobs': jobs}
    if name == 'replay':
        from tefed.replay import FrozenPool
        settings = config.get('replay', {})
        pool = FrozenPool(settings['pool_manifest'])
        protocol = _read(settings['exposure_protocol'])
        body = {key: value for key, value in protocol.items() if key != 'protocol_sha256'}
        if protocol.get('protocol_sha256') != canonical_hash(body):
            raise ValueError('Replay exposure protocol integrity check failed')
        for job in manifest['jobs']:
            job['patch']['replay'].update(expected_pool_sha256=pool.sha256,
                                         expected_protocol_sha256=protocol['protocol_sha256'])
        manifest.update(pool_sha256=pool.sha256, exposure_protocol_sha256=protocol['protocol_sha256'])
    for job in manifest['jobs']:
        resolved = _job_config(config, manifest, job)
        validate_run_config(resolved)
        validate_factorial_config(resolved)
    local = config.get('communication', {}).get('mode', 'local') == 'local'
    data = load_manifest(config['experiment']['manifest'], strict_files=local)
    binding = None
    scopes = {}
    for job in manifest['jobs']:
        resolved = _job_config(config, manifest, job)
        secret = source_secret(resolved)
        if binding is not None and sha256(secret).hexdigest() != binding['source_key_sha256']:
            raise ValueError('Suite source_key_sha256 changed while binding jobs')
        scope = _binding_scope(resolved)
        if scope in scopes:
            continue
        job_local = resolved.get('communication', {}).get('mode', 'local') == 'local'
        job_data = data if resolved['experiment']['manifest'] == config['experiment']['manifest'] else \
            load_manifest(resolved['experiment']['manifest'], strict_files=job_local)
        current = run_binding(resolved, job_data, labels_only=False, secret=secret,
                              workers=None if job_local else {})
        scopes[scope] = current
        if binding is None:
            binding = current
            continue
        for key in ('source_files', 'source_key_sha256', 'runtime'):
            if current[key] != binding[key]:
                raise ValueError(f'Suite {key} changed while binding jobs')
        for path, digest in current['input_files'].items():
            if path in binding['input_files'] and binding['input_files'][path] != digest:
                raise ValueError(f'Suite input changed while binding jobs: {path}')
            binding['input_files'][path] = digest
    manifest.update(suite_schema_version=1, suite_plan=name, binding=binding,
                    runner_sha256=file_hash(__file__))
    for job in manifest['jobs']:
        resolved = _job_config(config, manifest, job)
        if name == 'replay':
            from tefed.replay import ReplayController
            clients = sorted({str(record.client_id) for record in data.by_split('labeled')})
            ReplayController(resolved, clients, int(config['experiment'].get('rounds', 30)), job['seed'])
        job['configuration_sha256'] = canonical_hash({key: value for key, value in resolved.items()
                                                     if key != '_config_path'})
    manifest['suite_sha256'] = canonical_hash(manifest)
    return manifest


def _inspect_job(job, manifest, total_rounds, committed=None):
    from tefed.experiments import canonical_hash
    from tefed.resume import file_hash, load_snapshot
    output = Path(job['output'])
    if output.is_symlink() or output.resolve() != output:
        raise ValueError(f"Unsafe job directory: {job['variant']} seed {job['seed']}")
    if not output.exists():
        if committed is not None:
            raise ValueError(f"Missing completed job: {job['variant']} seed {job['seed']}")
        return None
    next_round = '?'
    try:
        snapshot_path = output / 'resume.pt'
        snapshot = load_snapshot(snapshot_path)
        next_round = snapshot.get('next_round')
        if snapshot.get('resume_schema_version') != 1 or snapshot.get('output_dir') != str(output):
            raise ValueError('Snapshot identity differs')
        if type(next_round) is not int or not 0 <= next_round <= total_rounds:
            raise ValueError('Snapshot round boundary is invalid')
        completed = snapshot.get('phase') == 'completed'
        if snapshot.get('phase') not in {'training', 'completed'} or completed and next_round != total_rounds:
            raise ValueError('Snapshot completion state is invalid')
        if [row['round_index'] for row in snapshot['history']] != list(range(next_round)):
            raise ValueError('Snapshot round history is not contiguous')
        configuration = _read(output / 'configuration.json')
        configuration.pop('_config_path', None)
        if canonical_hash(configuration) != job['configuration_sha256']:
            raise ValueError('Job configuration differs')
        binding = snapshot['binding']
        if binding['protocol_sha256'] != job['configuration_sha256']:
            raise ValueError('Snapshot configuration binding differs')
        for key in ('source_files', 'source_key_sha256', 'runtime'):
            if binding[key] != manifest['binding'][key]:
                raise ValueError(f'Snapshot {key} differs')
        if any(manifest['binding']['input_files'].get(path) != digest
               for path, digest in binding['input_files'].items()):
            raise ValueError('Snapshot input binding differs')
        for relative, record in snapshot['files'].items():
            path = (output / relative).resolve()
            if not path.is_relative_to(output) or not path.is_file():
                raise ValueError(f'Missing committed artifact: {relative}')
            size = path.stat().st_size
            if size < record['bytes'] or (completed or not record['append']) and size != record['bytes'] \
                    or file_hash(path, record['bytes']) != record['sha256']:
                raise ValueError(f'Committed artifact changed: {relative}')
        ledger = snapshot['files'].get('rounds.jsonl')
        rows = [] if ledger is None else (output / 'rounds.jsonl').read_bytes()[:ledger['bytes']].splitlines()
        if [json.loads(line)['round_index'] for line in rows] != list(range(next_round)):
            raise ValueError('Committed round ledger is not contiguous')
        if not completed:
            if committed is not None:
                raise ValueError('Completed job has a training snapshot')
            return {'resume': str(snapshot_path), 'next_round': next_round}
        required = {'final.pt', 'metrics.json', 'final.pool.json', 'source_states.json'}
        if not required.issubset(snapshot['files']):
            raise ValueError('Completed snapshot lacks result artifacts')
        metrics = _read(output / 'metrics.json')
        if canonical_hash(metrics) != canonical_hash(snapshot['metrics']):
            raise ValueError('Snapshot and saved metrics differ')
        result = {'metrics': metrics, 'next_round': next_round, 'snapshot_sha256': file_hash(snapshot_path)}
        if committed is not None and committed != result:
            raise ValueError('Completed result differs from the committed suite state')
        return {'result': result, 'next_round': next_round}
    except Exception as error:
        raise ValueError(f"{job['variant']} seed {job['seed']} round {next_round}: {error}") from error


def _publish_factorial(root, manifest, execution):
    if manifest['suite_plan'] != 'factorial':
        return
    records = {'factorial_runs.jsonl': []}
    for job, entry in zip(manifest['jobs'], execution['jobs']):
        if entry['status'] != 'completed':
            continue
        metrics = entry['result']['metrics']
        records['factorial_runs.jsonl'].append({'seed': job['seed'], 'factors': job['factors'],
            'protocol_sha256': manifest['protocol_sha256'], 'output': job['output'],
            'metrics': metrics, 'status': 'completed'})
        for view, values in metrics.items():
            for metric, value in values.items():
                if metric in ('mAP50', 'AP75', 'mAP50_95'):
                    records.setdefault(f'factorial_{view}_{metric}.jsonl', []).append({
                        'seed': job['seed'], 'factors': job['factors'], 'value': value,
                        'protocol_sha256': manifest['protocol_sha256']})
    for name, rows in records.items():
        _atomic_text(root / name, ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def run_suite(config, name, output, *, execute=False, resume=False, runner=None):
    from tefed.engine import run_experiment
    from tefed.experiments import FACTORIAL_DEFINITIONS
    from tefed.resume import run_lock
    root = Path(output).expanduser().resolve()
    if resume and config.get('communication', {}).get('mode', 'local') != 'local':
        raise ValueError('Suite resume requires communication.mode=local')
    with run_lock(root):
        manifest = make_manifest(config, name, root)
        if resume:
            if not (root / 'plan.json').is_file():
                raise FileNotFoundError('Suite resume requires the original plan.json')
            if _read(root / 'plan.json') != manifest:
                raise ValueError('Suite plan, configuration, seeds, operations or input binding changed')
            if name == 'factorial' and _read(root / 'operation_definitions.json') != FACTORIAL_DEFINITIONS:
                raise ValueError('Suite operation definitions changed')
            execution = _read(root / 'execution.json') if (root / 'execution.json').is_file() else None
            if execution is None and any(Path(job['output']).exists() for job in manifest['jobs']):
                raise ValueError('Existing jobs require the original execution.json history')
        else:
            if root.exists() and any(root.iterdir()):
                raise FileExistsError('Suite output already exists; use --resume or another directory')
            execution = None
        if execution is None:
            execution = {'suite_sha256': manifest['suite_sha256'], 'status': 'planned', 'history': [],
                         'jobs': [{'variant': job['variant'], 'seed': job['seed'], 'output': job['output'],
                                   'status': 'pending', 'attempts': []} for job in manifest['jobs']]}
        if execution.get('suite_sha256') != manifest['suite_sha256'] \
                or len(execution.get('jobs', [])) != len(manifest['jobs']) \
                or not isinstance(execution.get('history'), list) \
                or execution.get('status') not in {'planned', 'running', 'failed', 'completed'}:
            raise ValueError('Suite execution identity differs')
        inspections = []
        for job, entry in zip(manifest['jobs'], execution['jobs']):
            if any(entry.get(key) != job[key] for key in ('variant', 'seed', 'output')) \
                    or entry.get('status') not in {'pending', 'running', 'failed', 'completed'} \
                    or not isinstance(entry.get('attempts'), list) \
                    or any(attempt.get('status') not in {'running', 'failed', 'interrupted', 'completed'}
                           for attempt in entry['attempts']) \
                    or entry['status'] == 'completed' and not isinstance(entry.get('result'), dict):
                raise ValueError(f"Invalid suite job state: {job['variant']} seed {job['seed']}")
            inspections.append(_inspect_job(job, manifest, int(config['experiment'].get('rounds', 30)),
                                           entry.get('result') if entry['status'] == 'completed' else None))
        # Commit only after every existing job has passed read-only validation.
        root.mkdir(parents=True, exist_ok=True)
        if not resume:
            _write(root / 'plan.json', manifest)
            if name == 'factorial':
                _write(root / 'operation_definitions.json', FACTORIAL_DEFINITIONS)
        print(json.dumps({'jobs': len(manifest['jobs']), 'execute': execute,
                          'resume': resume, 'plan_file': str(root / 'plan.json')}, indent=2))
        if not execute:
            return execution
        execution['status'] = 'running'
        execution['history'].append({'event': 'resume' if resume else 'start', 'at': _timestamp()})
        _write(root / 'execution.json', execution)
        for job, entry, inspection in zip(manifest['jobs'], execution['jobs'], inspections):
            if inspection and 'result' in inspection:
                if entry['status'] != 'completed':
                    if entry['attempts'] and entry['attempts'][-1]['status'] == 'running':
                        entry['attempts'][-1].update(status='completed', finished_at=_timestamp())
                    entry.update(status='completed', result=inspection['result'])
                    _write(root / 'execution.json', execution)
                _publish_factorial(root, manifest, execution)
                continue
            if entry['attempts'] and entry['attempts'][-1]['status'] == 'running':
                entry['attempts'][-1].update(status='interrupted', finished_at=_timestamp())
            attempt = {'status': 'running', 'started_at': _timestamp(),
                       'next_round': inspection['next_round'] if inspection else 0}
            entry['attempts'].append(attempt)
            entry['status'] = 'running'
            _write(root / 'execution.json', execution)
            try:
                kwargs = {'seed': job['seed'], 'output_dir': job['output']}
                if inspection:
                    kwargs['resume'] = inspection['resume']
                (runner or run_experiment)(_job_config(config, manifest, job), **kwargs)
                checked = _inspect_job(job, manifest, int(config['experiment'].get('rounds', 30)))
                if not checked or 'result' not in checked:
                    raise ValueError('Training returned without a completed round snapshot')
                attempt.update(status='completed', finished_at=_timestamp())
                entry.update(status='completed', result=checked['result'])
                _write(root / 'execution.json', execution)
                _publish_factorial(root, manifest, execution)
            except BaseException as error:
                attempt.update(status='failed', finished_at=_timestamp(), error=str(error))
                entry.update(status='failed')
                execution['status'] = 'failed'
                _write(root / 'execution.json', execution)
                raise RuntimeError(f"{job['variant']} seed {job['seed']} round {attempt['next_round']}: {error}") from error
        execution['status'] = 'completed'
        _write(root / 'execution.json', execution)
        return execution


def main():
    from tefed.engine import load_config
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--plan',choices=['comparisons','mechanisms','budgets','components','factorial','replay'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true',help='Validate the existing suite and continue unfinished jobs')
    args=p.parse_args()
    run_suite(load_config(args.config),args.plan,args.output,execute=args.execute,resume=args.resume)


if __name__=='__main__':
    main()
