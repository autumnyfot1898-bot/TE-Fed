from copy import deepcopy
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from unittest.mock import patch
import zipfile

import numpy as np

from tefed.transport import pack_payload, unpack_payload, payload_digest, shared_config_digest, RemoteWorker


def write_toml(path, value):
    lines = []
    for key, item in value.items():
        if not isinstance(item, dict):
            lines.append(f'{key} = {json.dumps(item)}')
    for key, item in value.items():
        if isinstance(item, dict):
            lines.append(f'[{key}]')
            for name, setting in item.items():
                lines.append(f'{name} = {json.dumps(setting)}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


class TransportTests(unittest.TestCase):
    def test_numeric_roundtrip_preserves_scalar_dtype_and_tuples(self):
        value = {'float': np.arange(6, dtype=np.float32).reshape(2, 3),
                 'counter': np.array(3, dtype=np.int64), 'rng': (1, np.array([2], dtype=np.uint32), None)}
        restored = unpack_payload(pack_payload(value))
        self.assertEqual(restored['counter'].shape, ())
        self.assertEqual(restored['counter'].dtype, np.dtype('int64'))
        self.assertIsInstance(restored['rng'], tuple)
        self.assertEqual(payload_digest(value), payload_digest(restored))

    def test_object_arrays_and_nonfinite_values_are_rejected(self):
        for value in (np.array([object()], dtype=object), np.array([np.nan]), float('inf')):
            with self.assertRaises(ValueError):
                pack_payload(value)

    def test_unknown_archive_members_are_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('../escape', 'invalid')
        with self.assertRaises(ValueError):
            unpack_payload(stream.getvalue())

    def test_config_digest_ignores_machine_paths_but_keeps_protocol(self):
        from tefed.engine import load_config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('server', 'worker'):
                write_toml(root / f'{name}.toml', {'experiment': {'manifest': f'{name}/data.json', 'device': 'cpu'}, 'loss': {'beta': .5}})
            left, right = (load_config(root / f'{name}.toml') for name in ('server', 'worker'))
            self.assertEqual(shared_config_digest(left), shared_config_digest(right))
            right['loss']['beta'] = 0.
            self.assertNotEqual(shared_config_digest(left), shared_config_digest(right))

    def test_nonloopback_plaintext_is_rejected(self):
        with self.assertRaises(ValueError):
            RemoteWorker('http://192.0.2.1:8765', b'x' * 32)

    def test_worker_source_key_digest_is_bound_without_plaintext(self):
        from tefed.software_fixture import create_fixture
        from tefed.worker import WorkerService
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = create_fixture(root / 'data')
            with patch('tefed.engine.source_secret', return_value=b'a' * 32):
                first = WorkerService(config, 'client0', 'node0', b't' * 32, root / 'cache0')
            with patch('tefed.engine.source_secret', return_value=b'b' * 32):
                second = WorkerService(config, 'client0', 'node0', b't' * 32, root / 'cache1')
                with self.assertRaisesRegex(ValueError, 'source identifier key changed'):
                    first.execute({'action': 'client_round'})
            self.assertEqual(first.info['source_key_sha256'], hashlib.sha256(b'a' * 32).hexdigest())
            self.assertNotEqual(first.info['source_key_sha256'], second.info['source_key_sha256'])
            self.assertNotIn(b'a' * 32, pack_payload(first.info))


class NativeRemoteTests(unittest.TestCase):
    def test_two_workers_match_local_tasks_without_coordinator_images(self):
        import torch
        from tefed.client import execute_client_task, capture_rng_state, restore_rng_state, module_state
        from tefed.data import load_manifest
        from tefed.engine import load_config, seed_everything, source_secret
        from tefed.model import RoutedYOLO, Router
        from tefed.software_fixture import create_fixture
        from tefed.transport import partition_digest
        from tefed.worker import WorkerService, make_server, RemoteClientExecutor
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'TEFED_TRANSPORT_KEY': 't' * 32}):
            root = Path(folder)
            reference_config = create_fixture(root / 'reference')
            reference = load_manifest(reference_config['experiment']['manifest'], strict_files=True)
            coordinator_config = create_fixture(root / 'coordinator')
            manifest_path = Path(coordinator_config['experiment']['manifest'])
            rows = json.loads(manifest_path.read_text())
            for row in rows['records']:
                if row['id'] not in rows['calibration_ids']:
                    for field in ('rgb', 'depth', 'nir'):
                        if field in row:
                            row[field] = str(root / 'unavailable-on-coordinator' / Path(row[field]).name)
            manifest_path.write_text(json.dumps(rows), encoding='utf-8')
            coordinator_path = root / 'server.toml'
            write_toml(coordinator_path, coordinator_config)
            coordinator_config = load_config(coordinator_path)
            manifest = load_manifest(manifest_path, strict_files=False)
            servers, threads = [], []
            try:
                worker_settings = {}
                for index in range(2):
                    client_id = f'client{index}'
                    config = create_fixture(root / f'worker{index}')
                    path = Path(config['experiment']['manifest'])
                    data = json.loads(path.read_text())
                    data['records'] = [row for row in data['records'] if row['client_id'] == client_id]
                    ids = {row['id'] for row in data['records']}
                    data['calibration_ids'] = [uid for uid in data['calibration_ids'] if uid in ids]
                    data['views'] = {key: [uid for uid in values if uid in ids] for key, values in data['views'].items()}
                    path.write_text(json.dumps(data), encoding='utf-8')
                    config_path = root / f'worker{index}.toml'
                    write_toml(config_path, config)
                    service = WorkerService(load_config(config_path), client_id, f'node{index}', b't' * 32, root / f'cache{index}')
                    server = make_server(service)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    servers.append(server); threads.append(thread)
                    worker_settings[client_id] = {'url': f'http://127.0.0.1:{server.server_address[1]}', 'worker_id': f'node{index}'}
                coordinator_config['communication'] = {'mode': 'http', 'workers': worker_settings}
                executor = RemoteClientExecutor(coordinator_config, manifest, 'native-loopback')
                with self.assertRaises(HTTPError):
                    RemoteWorker(worker_settings['client0']['url'], b'wrong-key' * 4).call({'action': 'describe', 'task_id': 'wrong-key'})
                calibration = json.loads(Path(reference_config['experiment']['calibration']).read_text())
                seed_everything(5)
                model, router = RoutedYOLO(coordinator_config), Router()
                base_model, base_router = payload_digest(module_state(model)), payload_digest(module_state(router))
                states, persistent = {}, {}
                case_dir = root / 'server-generated-case'
                case_dir.mkdir()
                from PIL import Image
                image = np.full((64, 64, 3), 70, dtype=np.uint8)
                image[16:48, 16:40] = [180, 45, 30]
                Image.fromarray(image).save(case_dir / 'rgb.png')
                np.save(case_dir / 'depth.npy', np.full((64, 64), 1.5, dtype=np.float32))
                case = {'id': 'server-generated-test-case', 'round': 0, 'fidelity': .9,
                    'rgb': str(case_dir / 'rgb.png'), 'depth': str(case_dir / 'depth.npy'), 'nir': None,
                    'depth_unit_conversion': 1., 'targets': [{'box': [.25, .25, .625, .75], 'class_id': 0}],
                    'focal_target_index': 0, 'render_metadata': {'data_origin': 'generated', 'server_generated': True}}
                for client_id in worker_settings:
                    rng = capture_rng_state()
                    local = execute_client_task(reference_config, reference, model, router, calibration, action='initialize', client_id=client_id)
                    restore_rng_state(rng)
                    remote = executor.execute('initialize', model, router, calibration, client_id=client_id)
                    self.assertEqual(local, remote)
                    states[client_id] = remote['state']
                from tefed.core import make_descriptor
                source=reference.by_split('unlabeled','client0')[0]
                case['d']=make_descriptor({'id':'render-source','class_id':0,'disposition':'correction',
                    'route':[0,1,2],'weights':[.5,.3,.2]},[.5]*8,[1]*8,[1]*5,
                    'client0',source.id,0,states['client0'],source_secret(reference_config))
                for round_index in range(2):
                    for client_id in worker_settings:
                        selected = [case] if round_index == 1 else []
                        rng = capture_rng_state()
                        local = execute_client_task(reference_config, reference, model, router, calibration, action='client_round', client_id=client_id,
                            round_index=round_index, state=states[client_id], selected_cases=selected, persistent_state=persistent.get(client_id), run_secret=source_secret(reference_config))
                        restore_rng_state(rng)
                        remote = executor.execute('client_round', model, router, calibration, client_id=client_id,
                            round_index=round_index, state=states[client_id], selected_cases=selected, persistent_state=persistent.get(client_id))
                        self.assertEqual(payload_digest(local), payload_digest(remote))
                        self.assertGreater(remote['ledger']['optimizer_steps'], 0)
                        self.assertTrue(remote['candidate_audit'])
                        if round_index == 1:
                            self.assertGreater(remote['ledger']['presentations']['correction'], 0)
                            event=remote['selected_events'][0]
                            self.assertEqual(event['descriptor_id'],case['d']['id'])
                            self.assertEqual(event['source_observation_key'],case['d']['source'])
                            self.assertEqual(event['actual_presentations'],remote['ledger']['presentations']['correction'])
                            restore_rng_state(rng)
                            restarted = RemoteClientExecutor(coordinator_config, manifest, 'native-loopback')
                            retried = restarted.execute('client_round', model, router, calibration, client_id=client_id,
                                round_index=round_index, state=states[client_id], selected_cases=selected, persistent_state=persistent.get(client_id))
                            self.assertEqual(payload_digest(remote), payload_digest(retried))
                        self.assertNotIn(str(root), json.dumps(remote, default=lambda value: 'numeric-array'))
                        states[client_id], persistent[client_id] = remote['next_state'], remote['persistent_state']
                for client_id in worker_settings:
                    local = execute_client_task(reference_config, reference, model, router, calibration, action='evaluate', client_id=client_id, state=states[client_id])
                    remote = executor.execute('evaluate', model, router, calibration, client_id=client_id, state=states[client_id])
                    self.assertEqual(local, remote)
                    self.assertEqual(set(remote['views']), {'participating', 'unseen', 'new-client', 'hard'})
                self.assertEqual(base_model, payload_digest(module_state(model)))
                self.assertEqual(base_router, payload_digest(module_state(router)))
                coordinator_config['loss']['beta'] = .25
                with self.assertRaises(ValueError):
                    executor.execute('evaluate', model, router, calibration, client_id='client0', state=states['client0'])
                coordinator_config['loss']['beta'] = .5
                worker_manifest = load_manifest(root / 'worker0' / 'manifest.json', strict_files=True)
                worker_record = worker_manifest.by_split('labeled', 'client0')[0]
                worker_image = worker_record.rgb
                original_digest = partition_digest(worker_manifest, 'client0')
                self.assertEqual(original_digest, executor.fingerprints['client0']['partition_sha256'])
                previous = worker_image.read_bytes()
                worker_image.write_bytes(previous + b'changed-input')
                try:
                    self.assertNotEqual(partition_digest(worker_manifest, 'client0'), original_digest)
                    with self.assertRaises(RuntimeError):
                        executor.execute('evaluate', model, router, calibration, client_id='client0', state=states['client0'])
                finally:
                    worker_image.write_bytes(previous)
                self.assertEqual(worker_image.read_bytes(), previous)
                self.assertEqual(partition_digest(worker_manifest, 'client0'), original_digest)
            finally:
                for server in servers:
                    server.shutdown(); server.server_close()
                for thread in threads:
                    thread.join(5)


if __name__ == '__main__':
    unittest.main()
