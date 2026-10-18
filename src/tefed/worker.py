"""Execute authenticated client tasks using worker-local data and configuration."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import ssl
import threading
import time

import numpy as np

from .transport import (PROTOCOL_VERSION, MAX_BODY, RemoteWorker, pack_payload, unpack_payload,
                        payload_digest, shared_config_digest, partition_digest,
                        logical_partition_digest, signature, execution_identity)


def _key(name):
    value = os.environ.get(name, '').encode('utf-8')
    if len(value) < 32:
        raise ValueError(f'{name} must contain at least 32 UTF-8 bytes')
    return value


def _variant_config(config, method, seed):
    from .baselines import policy_dict
    value = deepcopy(config)
    value['experiment']['method'] = method
    value['experiment']['seed'] = int(seed)
    if not policy_dict(method, 0, 0)['use_experts']:
        value['model']['specialists'] = 0
    return value


def _case_downlink(case):
    from PIL import Image
    metadata = {key: deepcopy(value) for key, value in case.items() if key not in {'rgb', 'depth', 'nir'}}
    arrays = {}
    for name in ('rgb', 'depth', 'nir'):
        path = case.get(name)
        if path:
            path = Path(path)
            arrays[name] = np.load(path, allow_pickle=False) if path.suffix.lower() == '.npy' else np.asarray(Image.open(path)).copy()
    if 'rgb' not in arrays:
        raise ValueError('Selected server case has no RGB input')
    return {'metadata': metadata, 'arrays': arrays}


def _stage_case(case, cache):
    from PIL import Image
    if set(case) != {'metadata', 'arrays'} or set(case['arrays']) - {'rgb', 'depth', 'nir'}:
        raise ValueError('Invalid server case payload')
    metadata, arrays = deepcopy(case['metadata']), case['arrays']
    if any(name in metadata for name in ('rgb', 'depth', 'nir')):
        raise ValueError('Server paths are not accepted by workers')
    folder = cache / payload_digest(case)
    folder.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(arrays['rgb'])
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Server case RGB must be uint8 HWC3')
    path = folder / 'rgb.png'
    Image.fromarray(rgb).save(path)
    metadata['rgb'] = str(path)
    for name in ('depth', 'nir'):
        if name in arrays:
            value = np.asarray(arrays[name])
            if value.ndim != 2 or value.shape != rgb.shape[:2]:
                raise ValueError('Server modalities must be registered')
            path = folder / f'{name}.npy'
            np.save(path, value, allow_pickle=False)
            metadata[name] = str(path)
        else:
            metadata[name] = None
    return metadata


class WorkerService:
    def __init__(self, config, client_id, worker_id, secret, cache_dir):
        import torch
        from .worker_cache import WorkerTaskCache
        from .engine import validate_manifest_config, validate_run_config, source_secret
        from .data import load_manifest
        from .baselines import METHOD_REGISTRY
        self.config = deepcopy(config)
        validate_run_config(self.config)
        self.source_key = source_secret(self.config)
        self.client_id, self.worker_id = str(client_id), str(worker_id)
        self.secret = secret.encode() if isinstance(secret, str) else secret
        if not isinstance(self.secret, bytes) or len(self.secret) < 32:
            raise ValueError('Transport key must contain at least 32 bytes')
        self.manifest = load_manifest(config['experiment']['manifest'], strict_files=True)
        validate_manifest_config(self.config, self.manifest)
        self.calibration = json.loads(Path(config['experiment']['calibration']).read_text(encoding='utf-8'))
        self.cache = Path(cache_dir).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.nonces = {}
        self.info = {'worker_id': self.worker_id, 'client_id': self.client_id, 'protocol_version': PROTOCOL_VERSION,
            'device_type': 'cuda' if str(config['experiment'].get('device', 'cpu')).startswith('cuda') else 'cpu',
            'cuda_device_count': torch.cuda.device_count(),
            'model_schema_version': 2, 'local_config_sha256': payload_digest(self.config),
            'partition_sha256': partition_digest(self.manifest, self.client_id),
            'logical_partition_sha256': logical_partition_digest(self.manifest, self.client_id),
            'calibration_sha256': payload_digest(self.calibration),
            'source_key_sha256': hashlib.sha256(self.source_key).hexdigest(),
            **execution_identity(),
            'config_sha256_by_method': {method: shared_config_digest(_variant_config(self.config, method, 0)) for method in METHOD_REGISTRY}}
        self.task_cache = WorkerTaskCache(self.cache / 'tasks.sqlite3', self.info, self.secret, self.source_key)

    def execute(self, task):
        from .client import (execute_client_task, load_module_state, capture_rng_state, restore_rng_state)
        from .model import RoutedYOLO, Router
        from .engine import device_for, source_secret
        if task.get('action') == 'describe':
            return deepcopy(self.info)
        if not hmac.compare_digest(source_secret(self.config), self.source_key):
            raise ValueError('Worker source identifier key changed; restart the worker')
        allowed = {'protocol_version', 'model_schema_version', 'run_id', 'task_id', 'worker_id', 'client_id',
                   'action', 'round_index', 'method', 'seed', 'config_sha256', 'partition_sha256',
                   'calibration_sha256', 'bank_sha256', 'model', 'router', 'state', 'selected_cases',
                   'persistent_state', 'rng_state', 'replay_control', 'package_version', 'execution_sha256', 'source_key_sha256'}
        if set(task) - allowed:
            raise ValueError('Unknown task fields')
        for field in ('run_id', 'task_id'):
            if not isinstance(task.get(field), str) or not task[field] or len(task[field]) > 128:
                raise ValueError(f'Invalid {field}')
        if task.get('protocol_version') != PROTOCOL_VERSION or task.get('model_schema_version') != 2:
            raise ValueError('Protocol or model schema differs')
        if task.get('worker_id') != self.worker_id or task.get('client_id') != self.client_id:
            raise ValueError('Task identity does not match this worker')
        if task.get('source_key_sha256') != self.info['source_key_sha256']:
            raise ValueError('Worker source identifier key binding differs')
        current_code = execution_identity()
        if task.get('package_version') != self.info['package_version'] or task.get('execution_sha256') != self.info['source_sha256'] or current_code['source_sha256'] != self.info['source_sha256']:
            raise ValueError('Worker execution code differs; restart with the matching package')
        if task.get('partition_sha256') != self.info['partition_sha256'] or task.get('calibration_sha256') != self.info['calibration_sha256']:
            raise ValueError('Task data binding differs')
        if partition_digest(self.manifest, self.client_id) != self.info['partition_sha256']:
            raise ValueError('Worker input files changed after startup')
        config = _variant_config(self.config, task['method'], task['seed'])
        if task.get('config_sha256') != shared_config_digest(config):
            raise ValueError('Task configuration differs')
        if payload_digest({'model': task['model'], 'router': task['router']}) != task.get('bank_sha256'):
            raise ValueError('Task model bank hash differs')
        digest = payload_digest(task)
        index = task['round_index']
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError('Invalid round index')

        def compute():
            model = RoutedYOLO({**config, 'model': {**config['model'], 'weights': ''}}).to(device_for(config))
            router = Router(config.get('decision', {}).get('routing_temperature', .5)).to(device_for(config))
            load_module_state(model, task['model'])
            load_module_state(router, task['router'])
            selected = [_stage_case(value, self.cache) for value in task.get('selected_cases', [])]
            restore_rng_state(task['rng_state'])
            output = execute_client_task(config, self.manifest, model, router, self.calibration,
                action=task['action'], client_id=self.client_id, round_index=index, state=task.get('state'),
                selected_cases=selected, persistent_state=task.get('persistent_state'), run_secret=self.source_key,
                replay_control=task.get('replay_control'))
            return {'output': output, 'rng_state': capture_rng_state(), 'worker': deepcopy(self.info)}

        return self.task_cache.execute(task['run_id'], task['task_id'], digest, task['action'], index, compute)


def make_server(service, host='127.0.0.1', port=0, *, cert_file=None, key_file=None):
    if host not in {'127.0.0.1', 'localhost', '::1'} and not (cert_file and key_file):
        raise ValueError('Non-loopback workers require TLS')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            timestamp, nonce = self.headers.get('X-TEFed-Time', ''), self.headers.get('X-TEFed-Nonce', '')
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if self.path != '/task' or not 0 < size <= MAX_BODY or abs(time.time() - int(timestamp)) > 300 or len(nonce) != 32:
                    raise ValueError('Invalid request envelope')
                body = self.rfile.read(size)
                if len(body) != size or not hmac.compare_digest(self.headers.get('X-TEFed-Signature', ''), signature(service.secret, timestamp, nonce, body)):
                    raise ValueError('Request authentication failed')
                with service.lock:
                    service.nonces = {key: value for key, value in service.nonces.items() if time.time() - value < 600}
                    if nonce in service.nonces:
                        raise ValueError('Request nonce was already used')
                    service.nonces[nonce] = time.time()
                    task = unpack_payload(body)
                    try:
                        result = {'ok': True, 'task_id': task.get('task_id'), 'result': service.execute(task)}
                    except Exception as error:
                        result = {'ok': False, 'task_id': task.get('task_id'), 'error': f'{type(error).__name__}: client task failed'}
                response = pack_payload(result)
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.tefed.task')
                self.send_header('Content-Length', str(len(response)))
                self.send_header('X-TEFed-Signature', signature(service.secret, timestamp, nonce, response))
                self.end_headers()
                self.wfile.write(response)
            except Exception:
                self.send_error(400, 'Invalid authenticated task')
    server = ThreadingHTTPServer((host, int(port)), Handler)
    if cert_file and key_file:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


class RemoteClientExecutor:
    def __init__(self, config, manifest, run_id):
        self.config, self.manifest, self.run_id = config, manifest, str(run_id)
        self.clients, self.fingerprints = {}, {}
        workers = config.get('communication', {}).get('workers', {})
        expected = sorted({str(row.client_id) for row in manifest.by_split('labeled')})
        if set(workers) != set(expected):
            raise ValueError('Configure exactly one worker per participating client')
        for client_id in expected:
            settings = workers[client_id]
            remote = RemoteWorker(settings['url'], _key(settings.get('key_env', settings.get('token_env', 'TEFED_TRANSPORT_KEY'))),
                                  timeout=settings.get('timeout', 600), ca_file=settings.get('ca_file'))
            info = remote.call({'action': 'describe', 'task_id': secrets.token_hex(16)})
            if info['client_id'] != client_id or info['worker_id'] != settings['worker_id']:
                raise ValueError('Worker identity differs from configuration')
            code = execution_identity()
            if info['package_version'] != code['package_version'] or info['source_sha256'] != code['source_sha256']:
                raise ValueError('Coordinator/worker execution code differs')
            if info['dependencies'] != code['dependencies'] and not settings.get('allow_dependency_mismatch', False):
                raise ValueError('Worker dependency versions differ; explicitly record allowed build differences')
            if info['logical_partition_sha256'] != logical_partition_digest(manifest, client_id):
                raise ValueError('Worker logical partition differs')
            import torch
            device_type = 'cuda' if str(config['experiment'].get('device', 'cpu')).startswith('cuda') else 'cpu'
            if info['device_type'] != device_type:
                raise ValueError('Coordinator and worker device_type must match')
            if device_type == 'cuda' and info['cuda_device_count'] != torch.cuda.device_count():
                raise ValueError('Exact remote RNG requires matching CUDA device counts')
            for field in ('local_config_sha256', 'partition_sha256'):
                if settings.get(field) is not None and settings[field] != info[field]:
                    raise ValueError(f'Worker {field} differs')
            self.clients[client_id] = remote
            self.fingerprints[client_id] = {'url': settings['url'], **info}

    def execute(self, action, model, router, calibration, *, client_id, round_index=0, state=None,
                selected_cases=(), persistent_state=None, replay_control=None):
        from .client import module_state, capture_rng_state, restore_rng_state
        client_id = str(client_id)
        info = self.fingerprints[client_id]
        method = self.config['experiment'].get('method', 'tefed')
        config_hash = shared_config_digest(self.config)
        if info['config_sha256_by_method'].get(method) != config_hash or info['calibration_sha256'] != payload_digest(calibration):
            raise ValueError('Coordinator/worker protocol or calibration differs')
        bank = {'model': module_state(model), 'router': module_state(router)}
        rng_state = capture_rng_state()
        if info['device_type'] == 'cpu':
            rng_state['cuda'] = []
        task = {'protocol_version': PROTOCOL_VERSION, 'model_schema_version': 2,
            'run_id': self.run_id, 'worker_id': info['worker_id'],
            'client_id': client_id, 'action': action, 'round_index': int(round_index), 'method': method,
            'seed': int(self.config['experiment'].get('seed', 0)), 'config_sha256': config_hash,
            'package_version': info['package_version'], 'execution_sha256': info['source_sha256'],
            'source_key_sha256': info['source_key_sha256'],
            'partition_sha256': info['partition_sha256'], 'calibration_sha256': info['calibration_sha256'],
            'bank_sha256': payload_digest(bank), **bank, 'state': state,
            'selected_cases': [_case_downlink(case) for case in selected_cases],
            'persistent_state': persistent_state, 'rng_state': rng_state, 'replay_control': replay_control}
        task['task_id'] = payload_digest(task)
        result = self.clients[client_id].call(task)
        if result['worker'] != {key: value for key, value in info.items() if key != 'url'}:
            raise ValueError('Worker input binding changed')
        if info['device_type'] == 'cpu':
            result['rng_state']['cuda'] = []
        restore_rng_state(result['rng_state'])
        return result['output']


def main(argv=None):
    from .engine import load_config
    parser = argparse.ArgumentParser(description='Run a TE-Fed client worker')
    parser.add_argument('--config', required=True)
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--worker-id', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--key-env', default='TEFED_TRANSPORT_KEY')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--cert-file')
    parser.add_argument('--key-file')
    args = parser.parse_args(argv)
    service = WorkerService(load_config(args.config), args.client_id, args.worker_id, _key(args.key_env), args.cache_dir)
    server = make_server(service, args.host, args.port, cert_file=args.cert_file, key_file=args.key_file)
    print(json.dumps({'worker_id': args.worker_id, 'client_id': args.client_id, 'address': server.server_address}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
