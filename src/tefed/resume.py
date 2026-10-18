"""Atomic round snapshots and recovery of generated run files."""
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch


def file_hash(path, limit=None):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining=limit
        while remaining is None or remaining>0:
            data=stream.read(1024*1024 if remaining is None else min(remaining,1024*1024))
            if not data:
                break
            digest.update(data)
            if remaining is not None:
                remaining-=len(data)
    return digest.hexdigest()


def _pack(value):
    if isinstance(value,np.ndarray):
        array=value.astype(np.int64) if value.dtype.kind=='u' and value.dtype.itemsize>1 else value
        return {'__tefed_ndarray__':torch.from_numpy(array.copy()),'dtype':value.dtype.str}
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:_pack(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(_pack(item) for item in value)
    if isinstance(value,Path):
        return str(value)
    return value


def _unpack(value):
    if isinstance(value,dict):
        if set(value)=={'__tefed_ndarray__','dtype'}:
            return value['__tefed_ndarray__'].numpy().astype(value['dtype'],copy=True)
        return {key:_unpack(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(_unpack(item) for item in value)
    return value


def atomic_snapshot(path,payload):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(_pack(payload),stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def load_snapshot(path):
    return _unpack(torch.load(path,map_location='cpu',weights_only=True))


@contextmanager
def run_lock(output):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.parent/('.'+output.name+'.lock')
    with lock.open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another process is using run directory {output}') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def locked_run(function):
    @wraps(function)
    def wrapped(config,seed=0,output_dir=None,**kwargs):
        output=Path(output_dir or Path(config.get('experiment',{}).get('output','runs/tefed'))/f'seed_{seed}').resolve()
        with run_lock(output):
            return function(config,seed=seed,output_dir=output,**kwargs)
    return wrapped


def run_binding(config,manifest,*,labels_only,secret,workers=None):
    protocol={key:value for key,value in config.items() if key!='_config_path'}
    serialized=json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False)
    paths={str(Path(config['experiment']['manifest']).resolve())}
    if workers is None:
        for record in manifest.records:
            paths.update(str(Path(path).resolve()) for path in (record.rgb,record.depth,record.nir) if path is not None)
    else:
        for record in manifest.records:
            if str(record.client_id)=='server' and record.id in manifest.calibration_ids:
                paths.update(str(Path(path).resolve()) for path in (record.rgb,record.depth,record.nir) if path is not None)
    if not labels_only:
        paths.add(str(Path(config['experiment']['calibration']).resolve()))
    initial=config.get('model',{}).get('initial_checkpoint')
    if initial:
        paths.add(str(Path(initial).resolve()))
    for section,key in (('replay','pool_manifest'),('replay','exposure_protocol')):
        value=config.get(section,{}).get(key)
        if value:
            paths.add(str(Path(value).resolve()))
    renderer=config.get('renderer',{})
    if config.get('experiment',{}).get('method') in {'tefed','generic_hc'} and not labels_only and renderer.get('enabled',True) and config.get('decision',{}).get('render_budget',1152)>0 and not config.get('replay'):
        from .rendering import BlenderRenderer
        backend=BlenderRenderer(renderer.get('blender','blender'),renderer.get('scene'),
            timeout=float(renderer.get('timeout',600)),samples=int(renderer.get('samples',8)),
            script_path=renderer.get('script_path'))
        if backend.scene_path is None:
            raise ValueError('Active rendering requires renderer.scene')
        paths.update(str(path) for path in (backend.executable,backend.scene_path))
        if renderer.get('scene_class_map') is None:
            paths.add(str(backend.script_path))
        else:
            paths.update(str(path) for path in (backend.generic_script_path,
                backend.generic_script_path.parent/'blender_render.py'))
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','ultralytics','numpy')}
    return {'protocol_sha256':hashlib.sha256(serialized.encode()).hexdigest(),'labels_only':bool(labels_only),
            'input_files':{path:file_hash(path) for path in sorted(paths)},
            'source_files':{path.name:file_hash(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
            'source_key_sha256':hashlib.sha256(secret).hexdigest(),'workers':workers,
            'runtime':{'versions':versions,'python':sys.version.split()[0],'threads':torch.get_num_threads(),
                       'cuda_devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}


def _append_paths(output):
    return [output/name for name in ('rounds.jsonl','monitoring.jsonl','selected_events.jsonl')]+list(output.glob('client_*_candidate_audit.jsonl'))


def file_journal(output,previous=None,round_index=None,completed=False):
    output=Path(output)
    journal=dict(previous or {})
    paths=[output/name for name in ('configuration.json','calibration.json','environment.json')]
    append_paths=_append_paths(output)
    paths+=append_paths
    if round_index is not None:
        paths.append(output/f'verification_round_{round_index:03d}.json')
        for directory in ('rendering','monitoring'):
            paths+=list((output/directory/f'round_{round_index:03d}').rglob('*'))
    if completed:
        paths+=[output/name for name in ('final.pt','final.pool.json','source_states.json','metrics.json','local_model_selection.json')]
        paths+=list((output/'predictions').rglob('*'))
    for path in paths:
        if path.is_file():
            relative=str(path.relative_to(output)).replace('\\','/')
            journal[relative]={'bytes':path.stat().st_size,'sha256':file_hash(path),'append':path in append_paths}
    return journal


def reconcile_files(output,journal,next_round,completed=False):
    """Validate committed bytes before rolling back an unfinished round."""
    output=Path(output).resolve()
    for relative,record in journal.items():
        path=(output/relative).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError(f'Missing or invalid committed run file: {relative}')
        size=path.stat().st_size
        expected=record['bytes']
        if size<expected or ((completed or not record['append']) and size!=expected) or file_hash(path,expected)!=record['sha256']:
            raise ValueError(f'Committed run file changed: {relative}')
    ledger=journal.get('rounds.jsonl')
    lines=[] if ledger is None else (output/'rounds.jsonl').read_bytes()[:ledger['bytes']].splitlines()
    if [json.loads(line)['round_index'] for line in lines]!=list(range(next_round)):
        raise ValueError('Committed round ledger is not contiguous')
    if completed:
        return
    for path in _append_paths(output):
        if path.is_file():
            record=journal.get(path.relative_to(output).as_posix())
            if record is None:
                path.unlink()
            elif path.stat().st_size>record['bytes']:
                with path.open('r+b') as stream:
                    stream.truncate(record['bytes'])
    for path in output.glob('verification_round_*.json'):
        if int(path.stem.rsplit('_',1)[1])>=next_round:
            path.unlink()
    for name in ('rendering','monitoring'):
        for path in (output/name).glob('round_*'):
            if int(path.name.rsplit('_',1)[1])>=next_round:
                if path.is_symlink() or not path.resolve().is_relative_to(output) or any(p.is_symlink() for p in path.rglob('*')):
                    raise ValueError(f'Unsafe unfinished run directory: {path}')
                shutil.rmtree(path)
    for name in ('final.pt','final.pool.json','source_states.json','metrics.json','local_model_selection.json','resume.pt.tmp'):
        path=output/name
        if path.is_file():
            path.unlink()
    predictions=output/'predictions'
    if predictions.exists():
        if predictions.is_symlink() or any(p.is_symlink() for p in predictions.rglob('*')):
            raise ValueError('Unsafe unfinished prediction directory')
        shutil.rmtree(predictions)
