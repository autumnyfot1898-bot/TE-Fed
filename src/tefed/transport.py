"""Authenticated client tasks with JSON and non-object NumPy arrays."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import io
import importlib.metadata
import json
import math
import secrets
import ssl
import time
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler, HTTPSHandler
import zipfile

import numpy as np

PROTOCOL_VERSION = 1
MAX_BODY = 512 * 1024 * 1024
MAX_ARRAYS = 20000


def execution_identity():
    from pathlib import Path
    from . import __version__
    root = Path(__file__).resolve().parent
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob('*.py'))}
    return {'package_version': __version__, 'source_sha256': payload_digest(sources),
            'dependencies': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'ultralytics', 'numpy')}}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def payload_digest(value):
    def describe(item):
        if isinstance(item, np.ndarray):
            array = np.array(item, copy=True, order='C')
            return {'array': array.dtype.str, 'shape': list(array.shape), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
        if isinstance(item, np.generic):
            return describe(item.item())
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError('Payload keys must be strings')
            return {'dict': [[key, describe(item[key])] for key in sorted(item)]}
        if isinstance(item, (list, tuple)):
            return {type(item).__name__: [describe(value) for value in item]}
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise ValueError(f'Unsupported payload value: {type(item).__name__}')
    return hashlib.sha256(canonical_json(describe(value))).hexdigest()


def pack_payload(value):
    arrays = {}
    def encode(item, depth=0):
        if depth > 64:
            raise ValueError('Payload is too deeply nested')
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or item.dtype.fields or item.dtype.kind not in 'biuf' or item.dtype.itemsize > 8:
                raise ValueError('Only numeric non-object arrays are supported')
            if not np.isfinite(item).all():
                raise ValueError('Arrays must be finite')
            if len(arrays) >= MAX_ARRAYS or item.nbytes > MAX_BODY:
                raise ValueError('Array limit exceeded')
            name = f'arrays/{len(arrays):06d}.npy'
            stream = io.BytesIO()
            np.save(stream, np.array(item, copy=True, order='C'), allow_pickle=False)
            arrays[name] = stream.getvalue()
            return {'type': 'array', 'file': name, 'sha256': hashlib.sha256(arrays[name]).hexdigest()}
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError('Payload keys must be strings')
            return {'type': 'dict', 'items': [[key, encode(item[key], depth + 1)] for key in sorted(item)]}
        if isinstance(item, (list, tuple)):
            return {'type': type(item).__name__, 'items': [encode(value, depth + 1) for value in item]}
        if item is None or isinstance(item, (str, bool, int)) or isinstance(item, float) and math.isfinite(item):
            return item
        raise ValueError('Payload contains an unsupported value')
    metadata = canonical_json({'wire_version': PROTOCOL_VERSION, 'root': encode(value)})
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('metadata.json', metadata)
        for name, body in arrays.items():
            archive.writestr(name, body)
    result = output.getvalue()
    if len(result) > MAX_BODY:
        raise ValueError('Payload exceeds the size limit')
    return result


def unpack_payload(body):
    if len(body) > MAX_BODY:
        raise ValueError('Payload exceeds the size limit')
    def reject_constant(value):
        raise ValueError(f'Invalid JSON number: {value}')
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or len(names) > MAX_ARRAYS + 1 or sum(member.file_size for member in members) > MAX_BODY:
            raise ValueError('Invalid archive size or duplicate names')
        if 'metadata.json' not in names or any(name != 'metadata.json' and not (name.startswith('arrays/') and name.endswith('.npy') and name[7:-4].isdigit()) for name in names):
            raise ValueError('Unexpected archive member')
        metadata = json.loads(archive.read('metadata.json'), parse_constant=reject_constant)
        if metadata.get('wire_version') != PROTOCOL_VERSION:
            raise ValueError('Unsupported wire protocol')
        used = {'metadata.json'}
        def decode(item, depth=0):
            if depth > 64:
                raise ValueError('Payload is too deeply nested')
            if not isinstance(item, dict):
                if item is None or isinstance(item, (str, bool, int)) or isinstance(item, float) and math.isfinite(item):
                    return item
                raise ValueError('Invalid scalar payload')
            kind = item.get('type')
            if kind == 'array':
                name = item['file']
                if name not in names or name in used:
                    raise ValueError('Invalid array reference')
                used.add(name)
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise ValueError('Array hash differs')
                header = io.BytesIO(data)
                if np.lib.format.read_magic(header) != (1, 0):
                    raise ValueError('Unsupported array header')
                shape, _, dtype = np.lib.format.read_array_header_1_0(header, max_header_size=16384)
                if len(shape) > 16 or dtype.hasobject or dtype.fields or dtype.kind not in 'biuf' or dtype.itemsize > 8:
                    raise ValueError('Invalid array header')
                if math.prod(shape) * dtype.itemsize != len(data) - header.tell():
                    raise ValueError('Array shape differs from its bounded data')
                array = np.load(io.BytesIO(data), allow_pickle=False, max_header_size=16384)
                if array.dtype.hasobject or array.dtype.fields or array.dtype.kind not in 'biuf' or array.dtype.itemsize > 8 or not np.isfinite(array).all():
                    raise ValueError('Invalid numeric array')
                return array
            if kind == 'dict':
                result = {}
                for key, value in item['items']:
                    if not isinstance(key, str) or key in result:
                        raise ValueError('Invalid dictionary key')
                    result[key] = decode(value, depth + 1)
                return result
            if kind in {'list', 'tuple'}:
                values = [decode(value, depth + 1) for value in item['items']]
                return tuple(values) if kind == 'tuple' else values
            raise ValueError('Unknown payload type')
        result = decode(metadata['root'])
        if used != set(names):
            raise ValueError('Unreferenced archive data')
        return result


def shared_config_digest(config):
    value = deepcopy(config)
    value.pop('_config_path', None)
    value.pop('communication', None)
    value.pop('replay', None)
    for key in ('manifest', 'calibration', 'output', 'device', 'seed', 'method'):
        value.get('experiment', {}).pop(key, None)
    value.get('model', {}).pop('initial_checkpoint', None)
    for key in ('blender', 'scene', 'script_path'):
        value.get('renderer', {}).pop(key, None)
    return payload_digest(value)


def partition_digest(manifest, client_id):
    rows = []
    for record in sorted((row for row in manifest.records if str(row.client_id) == str(client_id)), key=lambda row: row.id):
        row = {'id': record.id, 'client_id': str(record.client_id), 'split': record.split,
               'labels': [target.as_dict() for target in record.labels] if record.split != 'unlabeled' else []}
        for name in ('rgb', 'depth', 'nir'):
            path = getattr(record, name)
            if path is not None:
                with path.open('rb') as stream:
                    row[name + '_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            else:
                row[name + '_sha256'] = None
        rows.append(row)
    if not rows:
        raise ValueError('Worker partition is empty')
    ids = {row['id'] for row in rows}
    views = {str(key): sorted(str(value) for value in values if str(value) in ids)
             for key, values in manifest.metadata.get('views', {}).items()}
    return payload_digest({'records': rows, 'views': views,
                           'calibration_ids': sorted(value for value in manifest.calibration_ids if value in ids)})


def logical_partition_digest(manifest, client_id):
    records = sorted((row for row in manifest.records if str(row.client_id) == str(client_id)), key=lambda row: row.id)
    ids = {row.id for row in records}
    return payload_digest({'records': [{'id': row.id, 'client_id': str(row.client_id), 'split': row.split,
        'labels': [target.as_dict() for target in row.labels] if row.split != 'unlabeled' else []} for row in records],
        'views': {str(key): sorted(str(value) for value in values if str(value) in ids)
                  for key, values in manifest.metadata.get('views', {}).items()},
        'calibration_ids': sorted(value for value in manifest.calibration_ids if value in ids)})


def signature(secret, timestamp, nonce, body):
    material = f'{PROTOCOL_VERSION}\n{timestamp}\n{nonce}\n{hashlib.sha256(body).hexdigest()}'.encode()
    return hmac.new(secret, material, 'sha256').hexdigest()


class RemoteWorker:
    def __init__(self, url, secret, *, timeout=600, ca_file=None):
        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https'} or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {'', '/'}:
            raise ValueError('Invalid worker URL')
        if parsed.scheme == 'http' and parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Non-loopback worker connections require HTTPS')
        self.url = url.rstrip('/') + '/task'
        self.secret = secret.encode() if isinstance(secret, str) else secret
        if not isinstance(self.secret, bytes) or len(self.secret) < 32:
            raise ValueError('Transport key must contain at least 32 bytes')
        self.timeout = float(timeout)
        self.context = ssl.create_default_context(cafile=ca_file) if parsed.scheme == 'https' else None
        self.opener = build_opener(ProxyHandler({}), HTTPSHandler(context=self.context))

    def call(self, task):
        body = pack_payload(task)
        timestamp, nonce = str(int(time.time())), secrets.token_hex(16)
        request = Request(self.url, data=body, method='POST', headers={
            'Content-Type': 'application/vnd.tefed.task', 'X-TEFed-Time': timestamp,
            'X-TEFed-Nonce': nonce, 'X-TEFed-Signature': signature(self.secret, timestamp, nonce, body)})
        with self.opener.open(request, timeout=self.timeout) as response:
            data = response.read(MAX_BODY + 1)
            expected = signature(self.secret, timestamp, nonce, data)
            if not hmac.compare_digest(response.headers.get('X-TEFed-Signature', ''), expected):
                raise ValueError('Worker response authentication failed')
        result = unpack_payload(data)
        if not result.get('ok'):
            raise RuntimeError(f'Worker task failed: {result.get("error", "unknown error")}')
        if result.get('task_id') != task.get('task_id'):
            raise ValueError('Worker returned a different task identity')
        return result['result']
