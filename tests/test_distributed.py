"""Native coordinator training and recovery through two HTTP workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import torch

from tefed import engine
from tefed.resume import load_snapshot
from tefed.software_fixture import create_fixture
from tefed.worker import WorkerService,make_server
from tests import test_resume as resume_tests


class LostCoordinatorCommit(RuntimeError):
    pass


def write_config(path,config):
    lines=[]
    def section(value,prefix=()):
        if prefix:
            lines.append('['+'.'.join(prefix)+']')
        for key,item in value.items():
            if not isinstance(item,dict):
                lines.append(key+' = '+json.dumps(item))
        for key,item in value.items():
            if isinstance(item,dict):
                section(item,(*prefix,key))
    section(config)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


class DistributedEngineTests(unittest.TestCase):
    def test_full_http_run_and_precommit_recovery_match_local(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory,patch.dict(os.environ,{'TEFED_TRANSPORT_KEY':'t'*32}):
            root=Path(directory)
            local_config=create_fixture(root/'local-data')
            local_config['evaluation']['monitor_every']=1
            local=engine.run_experiment(local_config,seed=3,output_dir=root/'local-run')
            remote_config=create_fixture(root/'coordinator')
            remote_config['evaluation']['monitor_every']=1
            manifest_path=Path(remote_config['experiment']['manifest'])
            manifest=json.loads(manifest_path.read_text())
            for record in manifest['records']:
                for field in ('rgb','depth','nir'):
                    if field in record:
                        record[field]=str(root/'not-on-coordinator'/Path(record[field]).name)
                        self.assertFalse(Path(record[field]).exists())
            manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
            servers=[];threads=[];settings={}
            try:
                for index in range(2):
                    client=f'client{index}'
                    config=create_fixture(root/f'worker-data-{index}')
                    config['evaluation']['monitor_every']=1
                    path=Path(config['experiment']['manifest'])
                    data=json.loads(path.read_text())
                    data['records']=[record for record in data['records'] if record['client_id']==client]
                    identities={record['id'] for record in data['records']}
                    data['calibration_ids']=[uid for uid in data['calibration_ids'] if uid in identities]
                    data['views']={view:[uid for uid in values if uid in identities] for view,values in data['views'].items()}
                    path.write_text(json.dumps(data),encoding='utf-8')
                    service=WorkerService(config,client,f'node{index}',b't'*32,root/f'worker-cache-{index}')
                    server=make_server(service)
                    thread=threading.Thread(target=server.serve_forever,daemon=True)
                    thread.start();servers.append(server);threads.append(thread)
                    settings[client]={'url':f'http://127.0.0.1:{server.server_address[1]}','worker_id':f'node{index}'}
                remote_config['communication']={'mode':'http','workers':settings}
                remote=engine.run_experiment(remote_config,seed=3,output_dir=root/'http-run')
                output=root/'http-interrupted'
                original_snapshot=engine.atomic_snapshot
                original_train=engine.train_local
                actual_training=[]
                def lose_commit(path,payload):
                    if payload.get('phase')=='training' and payload.get('next_round')==1:
                        raise LostCoordinatorCommit('workers finished; coordinator snapshot not committed')
                    return original_snapshot(path,payload)
                def count_training(model,items,config,**kwargs):
                    actual_training.append(kwargs['round_id'])
                    return original_train(model,items,config,**kwargs)
                with patch('tefed.engine.train_local',side_effect=count_training):
                    with patch('tefed.engine.atomic_snapshot',side_effect=lose_commit):
                        with self.assertRaises(LostCoordinatorCommit):
                            engine.run_experiment(remote_config,seed=3,output_dir=output)
                    self.assertEqual(actual_training,[0,0])
                    self.assertEqual(load_snapshot(output/'resume.pt')['next_round'],0)
                    recovered=engine.run_experiment(remote_config,seed=3,output_dir=output,resume=output/'resume.pt')
                    self.assertEqual(actual_training,[0,0,1,1])
                reference=load_snapshot(Path(local['output_dir'])/'resume.pt')
                comparison=resume_tests.ResumeTests()
                for result in (remote,recovered):
                    snapshot=load_snapshot(Path(result['output_dir'])/'resume.pt')
                    for field in ('model','router','states','source_states','pool','previous_state','client_runtime','rng'):
                        comparison.assert_tree_equal(reference[field],snapshot[field])
                    self.assertEqual(result['metrics'],local['metrics'])
                    self.assertEqual(resume_tests.stable_ledger(result['rounds']),resume_tests.stable_ledger(local['rounds']))
                    self.assertEqual(set(result['metrics']),{'participating','unseen','new-client','hard'})
                    rows=[json.loads(line) for line in (Path(result['output_dir'])/'rounds.jsonl').read_text().splitlines()]
                    self.assertEqual([row['round_index'] for row in rows],[0,1])
                    for view in result['metrics']:
                        predictions=Path(result['output_dir'])/'predictions'/f'{view}.jsonl'
                        self.assertEqual(len(predictions.read_text().splitlines()),2)
                    environment=json.loads((Path(result['output_dir'])/'environment.json').read_text())
                    self.assertEqual(set(environment['workers']),{'client0','client1'})
                    self.assertTrue(all(info['partition_sha256'] for info in environment['workers'].values()))
                config_path=root/'coordinator.toml'
                write_config(config_path,remote_config)
                cli_output=root/'cli-evaluation'
                environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                result=subprocess.run([sys.executable,'-m','tefed.main','evaluate','--config',str(config_path),
                    '--checkpoint',str(output/'final.pt'),'--output',str(cli_output)],
                    capture_output=True,text=True,env=environment,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                self.assertEqual(json.loads((cli_output/'metrics.json').read_text()),local['metrics'])
                for view in local['metrics']:
                    self.assertEqual(len((cli_output/'predictions'/f'{view}.jsonl').read_text().splitlines()),2)
                with patch('tefed.engine.RoutedYOLO') as constructor:
                    repeated=engine.run_experiment(remote_config,seed=3,output_dir=output,resume=output/'resume.pt')
                    constructor.assert_not_called()
                    self.assertEqual(repeated['metrics'],local['metrics'])
            finally:
                for server in servers:
                    server.shutdown();server.server_close()
                for thread in threads:
                    thread.join(5)


if __name__=='__main__':
    unittest.main()
