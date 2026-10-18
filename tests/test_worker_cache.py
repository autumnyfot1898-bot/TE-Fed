"""Worker restart, transaction and cache integrity checks."""
from copy import deepcopy
from contextlib import closing
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from tefed.transport import payload_digest, shared_config_digest
from tefed.worker_cache import WorkerTaskCache


BINDING = {'worker_id': 'node0', 'client_id': 'client0', 'config': 'config-v1',
           'partition': 'partition-v1', 'code': 'code-v1', 'source_key_sha256': 'source-v1'}
SECRET, SOURCE = b'transport-secret-' * 3, b'source-secret-' * 3


def _process_task(path, marker, ready, output, crash=False):
    try:
        cache = WorkerTaskCache(path, BINDING, SECRET, SOURCE)
        ready.wait(20)

        def compute():
            with open(marker, 'ab') as stream:
                stream.write(b'executed\n')
                stream.flush()
                os.fsync(stream.fileno())
            if crash:
                os._exit(23)
            time.sleep(.2)
            return {'values': np.arange(4, dtype=np.float32), 'rng': (7, None)}

        result = cache.execute('run', 'task0', 'digest0', 'client_round', 0, compute)
        output.put(('ok', payload_digest(result)))
    except Exception as error:
        output.put(('error', repr(error)))


class WorkerCacheTests(unittest.TestCase):
    def cache(self, path, **changes):
        return WorkerTaskCache(path, changes.get('binding', BINDING), changes.get('secret', SECRET), changes.get('source', SOURCE))

    def test_restart_keeps_results_older_than_two_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.sqlite3'
            cache = self.cache(path)
            first = {'values': np.arange(4, dtype=np.float32), 'rng': (7, None)}
            cache.execute('run', 'task0', 'digest0', 'client_round', 0, lambda: first)
            first['values'][0] = 100
            for index in range(1, 4):
                cache.execute('run', f'task{index}', f'digest{index}', 'client_round', index, lambda: {'round': index})
            restarted = self.cache(path)
            restored = restarted.execute('run', 'task0', 'digest0', 'client_round', 0, lambda: self.fail('Cached task executed again'))
            np.testing.assert_array_equal(restored['values'], np.arange(4, dtype=np.float32))
            self.assertIsInstance(restored['rng'], tuple)
            self.assertNotIn(SECRET, path.read_bytes())
            self.assertNotIn(SOURCE, path.read_bytes())

    def test_reused_identity_and_out_of_order_round_fail_before_compute(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.sqlite3'
            self.cache(path).execute('run', 'task0', 'digest0', 'client_round', 0, lambda: {'ok': True})
            cache = self.cache(path)
            with self.assertRaisesRegex(ValueError, 'different input'):
                cache.execute('run', 'task0', 'changed', 'client_round', 0, lambda: self.fail('Changed task executed'))
            with self.assertRaisesRegex(ValueError, 'advance exactly once'):
                cache.execute('run', 'task2', 'digest2', 'client_round', 2, lambda: self.fail('Skipped round executed'))

    def test_failed_execution_or_encoding_does_not_advance_round(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.sqlite3'
            cache = self.cache(path)
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                cache.execute('run', 'task0', 'digest0', 'client_round', 0, lambda: (_ for _ in ()).throw(RuntimeError('training failed')))
            with self.assertRaises(ValueError):
                cache.execute('run', 'task0', 'digest0', 'client_round', 0, lambda: {'invalid': np.array([np.nan])})
            self.cache(path).execute('run', 'task0', 'digest0', 'client_round', 0, lambda: {'ok': True})
            self.cache(path).execute('run', 'task1', 'digest1', 'client_round', 1, lambda: {'ok': True})

    def test_worker_config_data_code_or_key_changes_reject_existing_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.sqlite3'
            self.cache(path)
            for field in BINDING:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binding or authentication'):
                    self.cache(path, binding={**BINDING, field: 'changed'})
            for change in ({'secret': b'x' * 32}, {'source': b'y' * 32}):
                with self.assertRaisesRegex(ValueError, 'binding or authentication'):
                    self.cache(path, **change)

    def test_result_tampering_and_database_damage_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.sqlite3'
            self.cache(path).execute('run', 'task0', 'digest0', 'client_round', 0, lambda: {'ok': True})
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute('UPDATE results SET body=?', (b'changed',))
            with self.assertRaisesRegex(ValueError, 'result authentication'):
                self.cache(path).execute('run', 'task0', 'digest0', 'client_round', 0, lambda: self.fail('Damaged cache executed'))
            damaged = Path(folder) / 'damaged.sqlite3'
            damaged.write_bytes(b'not a database')
            with self.assertRaisesRegex(ValueError, 'database is damaged'):
                self.cache(damaged)

    def test_round_cursor_tampering_or_deletion_is_rejected(self):
        for command in ('UPDATE rounds SET round_index=10', 'DELETE FROM rounds', 'DELETE FROM results'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'tasks.sqlite3'
                self.cache(path).execute('run', 'task0', 'digest0', 'client_round', 0, lambda: {'ok': True})
                with closing(sqlite3.connect(path)) as connection, connection:
                    connection.execute(command)
                with self.assertRaisesRegex(ValueError, 'round cursor|authentication'):
                    self.cache(path).execute('run', 'task1', 'digest1', 'client_round', 1, lambda: self.fail('Damaged cursor executed'))

    def test_deleting_old_nonround_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.sqlite3'
            cache = self.cache(path)
            cache.execute('run', 'init', 'init-digest', 'initialize', 0, lambda: {'ok': True})
            cache.execute('run', 'task0', 'digest0', 'client_round', 0, lambda: {'ok': True})
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("DELETE FROM results WHERE task_id='init'")
            with self.assertRaisesRegex(ValueError, 'authentication'):
                self.cache(path)

    def test_two_processes_submit_one_completed_result(self):
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as folder:
            path, marker = str(Path(folder) / 'tasks.sqlite3'), str(Path(folder) / 'executions.txt')
            ready, output = context.Barrier(2), context.Queue()
            processes = [context.Process(target=_process_task, args=(path, marker, ready, output)) for _ in range(2)]
            try:
                for process in processes:
                    process.start()
                results = [output.get(timeout=40) for _ in processes]
                for process in processes:
                    process.join(40)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual([result[0] for result in results], ['ok', 'ok'], results)
                self.assertEqual(results[0][1], results[1][1])
                self.assertEqual(Path(marker).read_bytes(), b'executed\n')
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)
                output.close()

    def test_process_crash_rolls_back_uncommitted_result_and_cursor(self):
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as folder:
            path, marker = str(Path(folder) / 'tasks.sqlite3'), str(Path(folder) / 'executions.txt')
            ready, output = context.Event(), context.Queue()
            ready.set()
            process = context.Process(target=_process_task, args=(path, marker, ready, output, True))
            try:
                process.start()
                process.join(40)
                self.assertEqual(process.exitcode, 23)
                result = self.cache(path).execute('run', 'task0', 'digest0', 'client_round', 0, lambda: {'recomputed': True})
                self.assertTrue(result['recomputed'])
                self.cache(path).execute('run', 'task1', 'digest1', 'client_round', 1, lambda: {'ok': True})
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                output.close()


class NativeWorkerRestartTests(unittest.TestCase):
    def test_worker_restart_reuses_native_training_result(self):
        import torch
        from tefed.client import capture_rng_state, execute_client_task, module_state
        from tefed.engine import seed_everything
        from tefed.model import RoutedYOLO, Router
        from tefed.software_fixture import create_fixture
        from tefed.worker import WorkerService
        from tefed.transport import PROTOCOL_VERSION
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = create_fixture(root / 'data')
            service = WorkerService(config, 'client0', 'node0', SECRET, root / 'cache')
            seed_everything(7)
            model, router = RoutedYOLO(config), Router()
            bank = {'model': module_state(model), 'router': module_state(router)}
            task = {'protocol_version': PROTOCOL_VERSION, 'model_schema_version': 2,
                'run_id': 'restart-run', 'task_id': 'initialize', 'worker_id': 'node0', 'client_id': 'client0',
                'action': 'initialize', 'round_index': 0, 'method': 'tefed', 'seed': 7,
                'config_sha256': shared_config_digest(config), 'partition_sha256': service.info['partition_sha256'],
                'calibration_sha256': service.info['calibration_sha256'], 'bank_sha256': payload_digest(bank),
                'source_key_sha256': service.info['source_key_sha256'], 'package_version': service.info['package_version'],
                'execution_sha256': service.info['source_sha256'], 'rng_state': capture_rng_state(), **bank}
            task['state'] = service.execute(task)['output']['state']
            task.update(action='client_round', task_id='round0', rng_state=capture_rng_state())
            with patch('tefed.client.execute_client_task', wraps=execute_client_task) as training:
                original = service.execute(task)
                self.assertEqual(training.call_count, 1)
            self.assertGreater(original['output']['ledger']['optimizer_steps'], 0)
            restarted = WorkerService(config, 'client0', 'node0', SECRET, root / 'cache')
            with patch('tefed.client.execute_client_task', side_effect=AssertionError('Repeated training')):
                cached = restarted.execute(task)
            self.assertEqual(payload_digest(original), payload_digest(cached))
            changed = deepcopy(task)
            changed['seed'] = 8
            with self.assertRaisesRegex(ValueError, 'different input'):
                restarted.execute(changed)


if __name__ == '__main__':
    unittest.main()
