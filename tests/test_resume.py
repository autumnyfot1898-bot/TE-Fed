"""Round-boundary recovery against uninterrupted native training."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tefed import engine
from tefed.resume import file_journal,load_snapshot,reconcile_files,run_binding,run_lock
from tefed.software_fixture import create_fixture


class InterruptedRun(RuntimeError):
    pass


def stable_ledger(rows):
    return [{key:value for key,value in row.items() if key not in
             {'seconds','start_ts','end_ts','server_gpu_peak_bytes','monitoring_seconds'}} for row in rows]


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tree_equal(self,left,right):
        if isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for key in left:
                self.assert_tree_equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):
                self.assert_tree_equal(a,b)
        elif isinstance(left,torch.Tensor):
            self.assertTrue(torch.equal(left,right))
        elif isinstance(left,np.ndarray):
            self.assertTrue(np.array_equal(left,right))
        else:
            self.assertEqual(left,right)

    def compare_interrupted(self,root,method,device='cpu'):
        config=create_fixture(root/'fixture')
        config['experiment']['method']=method
        config['experiment']['device']=device
        config['optimizer']['amp']=device.startswith('cuda')
        config['optimizer']['amp_init_scale']=128.
        config['evaluation']['monitor_every']=1
        continuous=engine.run_experiment(config,output_dir=root/'continuous')
        output=root/'interrupted'
        original=engine.train_local
        def interrupt(model,items,configuration,**kwargs):
            result=original(model,items,configuration,**kwargs)
            if kwargs.get('round_id')==1:
                raise InterruptedRun('interrupt after a partial next round')
            return result
        with patch('tefed.engine.train_local',side_effect=interrupt):
            with self.assertRaises(InterruptedRun):
                engine.run_experiment(config,output_dir=output)
        snapshot=load_snapshot(output/'resume.pt')
        self.assertEqual(snapshot['next_round'],1)
        self.assertEqual(snapshot['phase'],'training')
        for folder in ('rendering','monitoring'):
            unfinished=output/folder/'round_001'
            unfinished.mkdir(parents=True,exist_ok=True)
            (unfinished/'partial.txt').write_text('unfinished')
        (output/'verification_round_001.json').write_text('[]')
        with (output/'rounds.jsonl').open('ab') as stream:
            stream.write(b'{"round_index":1,"partial":')
        resumed=engine.run_experiment(config,output_dir=output,resume=output/'resume.pt')
        full=load_snapshot(Path(continuous['output_dir'])/'resume.pt')
        recovered=load_snapshot(output/'resume.pt')
        for name in ('model','router','states','source_states','pool','previous_state','client_runtime','local_models','bn_cache','rng'):
            self.assert_tree_equal(full[name],recovered[name])
        self.assertEqual(continuous['metrics'],resumed['metrics'])
        self.assertEqual(stable_ledger(continuous['rounds']),stable_ledger(resumed['rounds']))
        rows=[json.loads(line) for line in (output/'rounds.jsonl').read_text().splitlines()]
        self.assertEqual([row['round_index'] for row in rows],[0,1])
        self.assertFalse((output/'rendering'/'round_001'/'partial.txt').exists())
        self.assertFalse((output/'monitoring'/'round_001'/'partial.txt').exists())
        with patch('tefed.engine.RoutedYOLO') as constructor:
            repeated=engine.run_experiment(config,output_dir=output,resume=output/'resume.pt')
            constructor.assert_not_called()
            self.assertEqual(repeated['metrics'],resumed['metrics'])
        self.assertEqual((output/'rounds.jsonl').read_text().count('"round_index"'),2)
        changed=deepcopy(config)
        changed['loss']['gamma']=.123
        with self.assertRaisesRegex(ValueError,'Resume protocol'):
            engine.run_experiment(changed,output_dir=output,resume=output/'resume.pt')
        return recovered

    def test_cpu_resume_preserves_aggregation_and_persistent_client_states(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for method in ('tefed','consistent_teacher_fedawa','local','fedbn'):
                with self.subTest(method=method):
                    snapshot=self.compare_interrupted(root/method,method)
                    if method=='consistent_teacher_fedawa':
                        self.assertTrue(all(state.get('teacher') for state in snapshot['client_runtime'].values()))
                    if method=='local':
                        self.assertEqual(len(snapshot['local_models']),2)
                    if method=='fedbn':
                        self.assertEqual(len(snapshot['bn_cache']),2)
                    if method=='tefed':
                        self.assertIsNotNone(snapshot['previous_state'])

    @unittest.skipUnless(torch.cuda.is_available(),'Requires a CUDA device')
    def test_cuda_amp_resume_is_tensor_identical(self):
        with TemporaryDirectory() as directory:
            self.compare_interrupted(Path(directory),'tefed','cuda:0')

    def test_changed_committed_files_are_rejected_before_truncation(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)
            path=output/'rounds.jsonl'
            path.write_text('{"round_index":0}\n')
            journal=file_journal(output)
            path.write_text('{"round_index":9}\nextra')
            with self.assertRaisesRegex(ValueError,'Committed run file changed'):
                reconcile_files(output,journal,1)
            self.assertIn('extra',path.read_text())

    def test_completed_ledger_rejects_appended_rows(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)
            path=output/'rounds.jsonl'
            path.write_text('{"round_index":0}\n')
            journal=file_journal(output)
            with path.open('a') as stream:
                stream.write('{"round_index":1}\n')
            with self.assertRaisesRegex(ValueError,'Committed run file changed'):
                reconcile_files(output,journal,1,completed=True)

    def test_concurrent_run_is_rejected(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'run'
            with run_lock(output):
                with self.assertRaisesRegex(RuntimeError,'Another process'):
                    with run_lock(output):
                        self.fail('Concurrent run lock was acquired')

    def test_remote_pretrain_fails_before_output(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            config['communication']={'mode':'http'}
            output=root/'must-not-exist'
            with self.assertRaisesRegex(ValueError,'pretrain requires communication.mode=local'):
                engine.pretrain(config,output_dir=output)
            self.assertFalse(output.exists())


class RendererBindingTests(unittest.TestCase):
    def binding(self,config,labels_only=False):
        from tefed.data import load_manifest
        manifest=load_manifest(config['experiment']['manifest'])
        return run_binding(config,manifest,labels_only=labels_only,secret=b'fixture-key')

    def test_default_and_override_renderer_scripts_are_bound(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            scene=root/'scene.blend';scene.write_bytes(b'scene')
            config['renderer']={'blender':sys.executable,'scene':str(scene),'enabled':True}
            config['decision']['render_budget']=1
            baseline=self.binding(config)
            default=Path(engine.__file__).resolve().parents[2]/'scripts'/'blender_render.py'
            self.assertIn(str(default),baseline['input_files'])
            self.assertIn(str(Path(sys.executable).resolve()),baseline['input_files'])
            custom=root/'custom_render.py';custom.write_text('first version')
            config['renderer']['script_path']=str(custom)
            baseline=self.binding(config)
            self.assertIn(str(custom),baseline['input_files'])
            custom.write_text('second version')
            self.assertNotEqual(baseline['input_files'][str(custom)],self.binding(config)['input_files'][str(custom)])

    def test_generic_script_shared_module_and_executable_are_bound(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            config['experiment']['method']='generic_hc'
            config['renderer']={'enabled':True,'scene_class_map':{'object:test':0}}
            config['decision']['render_budget']=1
            executable=root/'blender.exe';scene=root/'scene.blend'
            generic=root/'blender_generic_render.py';common=root/'blender_render.py'
            unused=root/'unused_script.py'
            for path in (executable,scene,generic,common,unused):
                path.write_bytes(path.name.encode())
            backend=SimpleNamespace(executable=executable,scene_path=scene,
                script_path=unused,generic_script_path=generic)
            with patch('tefed.rendering.BlenderRenderer',return_value=backend):
                baseline=self.binding(config)
                self.assertNotIn(str(unused),baseline['input_files'])
                for path in (executable,generic,common):
                    with self.subTest(path=path.name):
                        original=path.read_bytes()
                        path.write_bytes(original+b'changed')
                        self.assertNotEqual(baseline['input_files'][str(path)],self.binding(config)['input_files'][str(path)])
                        path.write_bytes(original)

    def test_inactive_render_paths_do_not_resolve_blender(self):
        with TemporaryDirectory() as directory:
            base=create_fixture(Path(directory)/'fixture')
            base['renderer']={'enabled':True,'blender':'missing-blender','scene':'missing.blend'}
            base['decision']['render_budget']=1
            variants=[]
            for method in ('fedavg','server_moe'):
                config=deepcopy(base);config['experiment']['method']=method
                variants.append((config,False))
            config=deepcopy(base);config['renderer']['enabled']=False;variants.append((config,False))
            config=deepcopy(base);config['decision']['render_budget']=0;variants.append((config,False))
            config=deepcopy(base);config['replay']={'arm':'matched'};variants.append((config,False))
            variants.append((base,True))
            for config,labels_only in variants:
                with self.subTest(config=config,labels_only=labels_only),patch('tefed.rendering.BlenderRenderer') as renderer:
                    self.binding(config,labels_only)
                    renderer.assert_not_called()


if __name__=='__main__':
    unittest.main()
