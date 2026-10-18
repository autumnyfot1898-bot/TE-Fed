"""Authenticated worker results committed with their round cursor."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
from pathlib import Path
import sqlite3
import threading

from .transport import canonical_json, pack_payload, payload_digest, unpack_payload


class WorkerTaskCache:
    def __init__(self, path, binding, secret, source_key):
        self.path = Path(path)
        self.binding = payload_digest(binding)
        self.key = hmac.new(secret, b'TE-Fed worker cache v1\0' + source_key, 'sha256').digest()
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._transaction() as connection:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables:
                    connection.execute('CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), binding TEXT NOT NULL, result_count INTEGER NOT NULL, round_count INTEGER NOT NULL, mac TEXT NOT NULL)')
                    connection.execute('CREATE TABLE results (run_id TEXT NOT NULL, task_id TEXT NOT NULL, digest TEXT NOT NULL, action TEXT NOT NULL, round_index INTEGER NOT NULL, body BLOB NOT NULL, mac TEXT NOT NULL, PRIMARY KEY(run_id, task_id))')
                    connection.execute('CREATE TABLE rounds (run_id TEXT PRIMARY KEY, round_index INTEGER NOT NULL, task_id TEXT NOT NULL, digest TEXT NOT NULL, mac TEXT NOT NULL, FOREIGN KEY(run_id, task_id) REFERENCES results(run_id, task_id))')
                    connection.execute('INSERT INTO metadata VALUES (1, ?, 0, 0, ?)', (self.binding, self._mac(['binding', self.binding, 0, 0])))
                elif tables != {'metadata', 'results', 'rounds'}:
                    raise ValueError('Worker cache schema differs')
                self._check_binding(connection)
                if connection.execute('PRAGMA quick_check').fetchone() != ('ok',):
                    raise ValueError('Worker cache is damaged')
        except sqlite3.DatabaseError as error:
            raise ValueError('Worker cache database is damaged or unavailable') from error

    @contextmanager
    def _transaction(self):
        with self.lock:
            connection = sqlite3.connect(self.path, timeout=3600, isolation_level=None)
            try:
                connection.execute('PRAGMA synchronous=FULL')
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute('BEGIN IMMEDIATE')
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _mac(self, value):
        return hmac.new(self.key, canonical_json(value), 'sha256').hexdigest()

    def _check_binding(self, connection):
        rows = connection.execute('SELECT id, binding, result_count, round_count, mac FROM metadata').fetchall()
        if len(rows) != 1 or rows[0][:2] != (1, self.binding):
            raise ValueError('Worker cache binding or authentication differs')
        _, binding, result_count, round_count, mac = rows[0]
        counts = (connection.execute('SELECT count(*) FROM results').fetchone()[0],
                  connection.execute('SELECT count(*) FROM rounds').fetchone()[0])
        if counts != (result_count, round_count) or not hmac.compare_digest(mac, self._mac(['binding', binding, result_count, round_count])):
            raise ValueError('Worker cache binding or authentication differs')
        return counts

    def _result_mac(self, run_id, task_id, digest, action, index, body):
        return self._mac(['result', self.binding, run_id, task_id, digest, action, index, hashlib.sha256(body).hexdigest()])

    def _read_result(self, connection, run_id, task_id):
        row = connection.execute('SELECT digest, action, round_index, body, mac FROM results WHERE run_id=? AND task_id=?', (run_id, task_id)).fetchone()
        if row is not None:
            digest, action, index, body, mac = row
            if not isinstance(body, bytes) or not hmac.compare_digest(mac, self._result_mac(run_id, task_id, digest, action, index, body)):
                raise ValueError('Worker cached result authentication failed')
        return row

    def _read_round(self, connection, run_id):
        row = connection.execute('SELECT round_index, task_id, digest, mac FROM rounds WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            if connection.execute("SELECT 1 FROM results WHERE run_id=? AND action='client_round' LIMIT 1", (run_id,)).fetchone():
                raise ValueError('Worker cache round cursor is missing')
            return None
        index, task_id, digest, mac = row
        expected = self._mac(['round', self.binding, run_id, index, task_id, digest])
        result = self._read_result(connection, run_id, task_id)
        if not hmac.compare_digest(mac, expected) or result is None or result[:3] != (digest, 'client_round', index):
            raise ValueError('Worker cache round cursor authentication failed')
        return index

    def execute(self, run_id, task_id, digest, action, index, compute):
        if action not in {'initialize', 'client_round', 'evaluate'}:
            raise ValueError('Unknown worker action')
        try:
            with self._transaction() as connection:
                result_count, round_count = self._check_binding(connection)
                previous = self._read_round(connection, run_id)
                cached = self._read_result(connection, run_id, task_id)
                if cached is not None:
                    if cached[:3] != (digest, action, index):
                        raise ValueError('Task identity was reused with different input')
                    return unpack_payload(cached[3])
                if action == 'client_round' and previous is not None and index != previous + 1:
                    raise ValueError('Client rounds must advance exactly once')
                result = compute()
                body = pack_payload(result)
                mac = self._result_mac(run_id, task_id, digest, action, index, body)
                connection.execute('INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)', (run_id, task_id, digest, action, index, body, mac))
                if action == 'client_round':
                    mac = self._mac(['round', self.binding, run_id, index, task_id, digest])
                    connection.execute('INSERT INTO rounds VALUES (?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET round_index=excluded.round_index, task_id=excluded.task_id, digest=excluded.digest, mac=excluded.mac', (run_id, index, task_id, digest, mac))
                    round_count += previous is None
                result_count += 1
                mac = self._mac(['binding', self.binding, result_count, round_count])
                connection.execute('UPDATE metadata SET result_count=?, round_count=?, mac=? WHERE id=1', (result_count, round_count, mac))
                return result
        except sqlite3.DatabaseError as error:
            raise ValueError('Worker cache transaction failed') from error
