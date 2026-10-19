"""Input policy and training preflight regressions."""
from copy import deepcopy
from hashlib import sha256
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
from tefed.data import load_manifest
from tefed.model import sample_tensor
from tefed.server import _load_case, update_router
from tefed.software_fixture import create_fixture
from tests.test_server import FakeBank, FakeRouter, calibration


class InputPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=TemporaryDirectory()
        cls.config=create_fixture(Path(cls.temp.name))
        cls.manifest=load_manifest(cls.config['experiment']['manifest'])
        cls.record=cls.manifest.by_split('validation')[0]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def case(self):
        record=self.record
        return {'id':record.id,'rgb':str(record.rgb),'depth':str(record.depth),
                'nir':str(record.nir),'depth_unit_conversion':1.,
                'targets':[{'box':list(a.box),'class_id':a.class_id} for a in record.labels]}

    def assert_same_inputs(self,left,right):
        self.assertEqual(left.get('enabled_modalities'),right.get('enabled_modalities'))
        self.assertTrue(torch.equal(sample_tensor(left),sample_tensor(right)))

    def test_rgb_only_rendered_and_dataset_inputs_match(self):
        config=deepcopy(self.config)
        config['model']['disable_auxiliary']=True
        regular=engine.prepared_sample(self.record,config)
        rendered=_load_case(self.case(),config)
        self.assert_same_inputs(regular,rendered)
        self.assertEqual(rendered['enabled_modalities'],['RGB'])
        self.assertEqual(float(sample_tensor(rendered)[:,3:].abs().sum()),0.)

    def test_drone_rendered_and_dataset_inputs_keep_ir_tag(self):
        config=deepcopy(self.config)
        config['task']='dronevehicle'
        regular=engine.prepared_sample(self.record,config)
        rendered=_load_case(self.case(),config)
        self.assert_same_inputs(regular,rendered)
        self.assertEqual(rendered['enabled_modalities'],['RGB','IR'])
        self.assertIsNone(rendered['depth'])
        self.assertIs(rendered['ir'],rendered['nir'])
        self.assertGreater(float(sample_tensor(rendered)[:,4:].sum()),0.)

    def test_policy_does_not_mutate_caller_arrays_or_tags(self):
        source={'rgb':np.zeros((32,32,3),dtype=np.uint8),
                'depth':np.ones((32,32)),'nir':np.ones((32,32)),
                'enabled_modalities':['RGB','D','NIR']}
        result=engine.apply_input_policy(source,{'model':{'disable_auxiliary':True}})
        self.assertIsNotNone(source['depth'])
        self.assertIsNotNone(source['nir'])
        self.assertEqual(source['enabled_modalities'],['RGB','D','NIR'])
        self.assertIsNone(result['depth'])
        self.assertIsNone(result['nir'])

    def test_router_calibration_uses_the_same_input_policy(self):
        manifest=SimpleNamespace(calibration_ids=(self.record.id,),by_id={self.record.id:self.record})
        for mode in ('rgb_only','dronevehicle'):
            config=deepcopy(self.config)
            if mode=='rgb_only':
                config['model']['disable_auxiliary']=True
            else:
                config['task']='dronevehicle'
            captured=[]
            def probe(sample,*args,**kwargs):
                captured.append(deepcopy(sample))
                return None,'test_probe'
            with self.subTest(mode=mode),patch('tefed.server._probe_sample',side_effect=probe):
                update_router(FakeBank(),FakeRouter(),manifest,[],{},
                              {self.record.client_id:{'q':.5,'p':.5,'h':[.2]*5}},calibration(),config,0)
            self.assertEqual(len(captured),3)
            regular=engine.prepared_sample(self.record,config)
            for sample in captured:
                self.assertEqual(sample['enabled_modalities'],regular['enabled_modalities'])
                self.assertTrue(torch.equal(sample_tensor(sample)[:,3:],sample_tensor(regular)[:,3:]))

    def test_assessment_masks_raw_auxiliary_inputs(self):
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.zeros(()))
                self.inputs=[]
                self.last_proposal_counts={0:[0]}
            def features(self,value):
                self.inputs.append(value.detach().clone())
                return []
            def predict_heads(self,value,*args,cached_features=None,**kwargs):
                if cached_features is None:
                    self.features(value)
                return {0:[[]]}
        model=Recorder()
        config=deepcopy(self.config)
        config['model']['disable_auxiliary']=True
        raw={'id':'raw','rgb':np.full((32,32,3),128,dtype=np.uint8),
             'depth':np.ones((32,32)),'nir':np.full((32,32),255,dtype=np.uint8),'targets':[]}
        engine.assess_frame(model,None,raw,{'q':.5,'p':.5,'h':[.2]*5},calibration(),config)
        self.assertTrue(model.inputs)
        self.assertTrue(all(float(value[:,3:].abs().sum())==0 for value in model.inputs))


class CalibrationBindingTests(unittest.TestCase):
    def test_matching_hash_is_accepted_and_mismatch_is_rejected(self):
        with TemporaryDirectory() as directory:
            initial=Path(directory)/'initial.pt'
            initial.write_bytes(b'checkpoint-a')
            config={'model':{'initial_checkpoint':str(initial)}}
            digest=sha256(initial.read_bytes()).hexdigest()
            engine.validate_calibration_binding(config,{'initial_checkpoint_sha256':digest})
            initial.write_bytes(b'checkpoint-b')
            with self.assertRaisesRegex(ValueError,'does not match'):
                engine.validate_calibration_binding(config,{'initial_checkpoint_sha256':digest})

    def test_measured_artifact_requires_a_checkpoint_and_hash(self):
        with self.assertRaisesRegex(ValueError,'model.initial_checkpoint'):
            engine.validate_calibration_binding({'model':{}},{})
        for value in (None,'short','x'*64):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'initial_checkpoint_sha256'):
                engine.validate_calibration_binding({'model':{'initial_checkpoint':'unused.pt'}},
                                                    {'initial_checkpoint_sha256':value})

    def test_fixture_calibration_requires_explicit_mode(self):
        engine.validate_calibration_binding({'experiment':{'test_mode':True}}, {'data_origin':'software_fixture'})
        for config,artifact in (({'experiment':{'test_mode':True}},{}),
                                ({'experiment':{'test_mode':False}},{'data_origin':'software_fixture'})):
            with self.subTest(config=config),self.assertRaisesRegex(ValueError,'model.initial_checkpoint'):
                engine.validate_calibration_binding(config,artifact)
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            config['experiment']['test_mode']=False
            with patch.dict('os.environ',{'TEFED_SOURCE_SECRET':'k'*32}),\
                    patch('tefed.engine.RoutedYOLO') as constructor:
                with self.assertRaisesRegex(ValueError,'Fixture calibration requires experiment.test_mode=true'):
                    engine.run_experiment(config,output_dir=root/'run')
                constructor.assert_not_called()
                self.assertFalse((root/'run').exists())

    def test_bad_binding_fails_before_output_or_model_construction(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            config['experiment']['test_mode']=False
            initial=root/'initial.pt'
            initial.write_bytes(b'checkpoint-a')
            config['model']['initial_checkpoint']=str(initial)
            artifact=json.loads(Path(config['experiment']['calibration']).read_text())
            artifact['data_origin']='measured'
            for expected in (None,sha256(b'other-checkpoint').hexdigest()):
                artifact['initial_checkpoint_sha256']=expected
                Path(config['experiment']['calibration']).write_text(json.dumps(artifact))
                output=root/'must-not-exist'
                with self.subTest(expected=expected),patch.dict('os.environ',{'TEFED_SOURCE_SECRET':'k'*32}),\
                        patch('tefed.engine.RoutedYOLO') as constructor:
                    with self.assertRaisesRegex(ValueError,'initial_checkpoint_sha256'):
                        engine.run_experiment(config,output_dir=output)
                    constructor.assert_not_called()
                    self.assertFalse(output.exists())

    def test_test_mode_requires_fixture_manifest_before_output(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            path=Path(config['experiment']['manifest'])
            manifest=json.loads(path.read_text())
            for origin in (None,'measured'):
                if origin is None:
                    manifest.pop('data_origin',None)
                else:
                    manifest['data_origin']=origin
                path.write_text(json.dumps(manifest))
                with self.subTest(origin=origin),patch('tefed.engine.RoutedYOLO') as constructor:
                    with self.assertRaisesRegex(ValueError,'test_mode requires manifest.data_origin=software_fixture'):
                        engine.run_experiment(config,output_dir=root/'run')
                    constructor.assert_not_called()
                    self.assertFalse((root/'run').exists())

    def test_model_class_mismatch_fails_before_output(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            path=Path(config['experiment']['manifest'])
            original=json.loads(path.read_text())
            for declared_names in (True,False):
                manifest=deepcopy(original)
                if declared_names:
                    manifest['class_names'].append('extra')
                    expected='model.classes must match manifest.class_names'
                else:
                    manifest.pop('class_names')
                    manifest['records'][0]['labels'][0]['class_id']=2
                    expected='class_id must be below model.classes'
                path.write_text(json.dumps(manifest))
                with self.subTest(declared_names=declared_names),patch('tefed.engine.RoutedYOLO') as constructor:
                    with self.assertRaisesRegex(ValueError,expected):
                        engine.run_experiment(config,output_dir=root/'run')
                    constructor.assert_not_called()
                    self.assertFalse((root/'run').exists())

    def test_pretraining_does_not_load_or_bind_calibration(self):
        with TemporaryDirectory() as directory:
            config=create_fixture(Path(directory)/'fixture')
            config['experiment']['test_mode']=False
            config['experiment']['calibration']='missing-calibration.json'
            config['renderer']['enabled']=True
            config['renderer']['blender']='missing-blender'
            with patch.dict('os.environ',{'TEFED_SOURCE_SECRET':'k'*32}),\
                    patch('tefed.engine.validate_calibration_binding') as binding,\
                    patch('tefed.engine.RoutedYOLO',side_effect=RuntimeError('stop at construction')):
                with self.assertRaisesRegex(RuntimeError,'stop at construction'):
                    engine.pretrain(config,output_dir=Path(directory)/'pretrain')
                binding.assert_not_called()


class RendererPreflightTests(unittest.TestCase):
    def test_inactive_correction_paths_do_not_require_blender(self):
        base={'renderer':{'enabled':True,'blender':'missing-blender','scene':'missing.blend'},
              'decision':{'render_budget':1}}
        disabled=deepcopy(base);disabled['renderer']['enabled']=False
        zero_budget=deepcopy(base);zero_budget['decision']['render_budget']=0
        for config,policy in ((base,{'use_corrections':False}),
                              (disabled,{'use_corrections':True}),
                              (zero_budget,{'use_corrections':True})):
            with self.subTest(config=config),patch('tefed.engine.BlenderRenderer') as constructor:
                engine.validate_renderer_setup(config,policy)
                constructor.assert_not_called()

    def test_missing_blender_fails_before_output_or_model_construction(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            config=create_fixture(root/'fixture')
            config['renderer']={'blender':str(root/'missing-blender.exe'),'scene':str(root/'scene.blend')}
            config['decision']['render_budget']=1
            output=root/'must-not-exist'
            with patch('tefed.engine.RoutedYOLO') as constructor:
                with self.assertRaisesRegex(FileNotFoundError,'Blender executable'):
                    engine.run_experiment(config,output_dir=output)
                constructor.assert_not_called()
                self.assertFalse(output.exists())

    def test_missing_scene_is_rejected(self):
        with TemporaryDirectory() as directory:
            config={'renderer':{'blender':sys.executable,'scene':str(Path(directory)/'missing.blend')}}
            with self.assertRaisesRegex(FileNotFoundError,'Blender scene'):
                engine.validate_renderer_setup(config,{'use_corrections':True})

    def test_absolute_and_path_executables_resolve_without_launching(self):
        with TemporaryDirectory() as directory:
            scene=Path(directory)/'scene.blend'
            scene.write_bytes(b'preflight-only')
            config={'renderer':{'blender':sys.executable,'scene':str(scene)}}
            engine.validate_renderer_setup(config,{'use_corrections':True})
            config['renderer']['blender']='test-blender-from-path'
            with patch('tefed.rendering.shutil.which',return_value=sys.executable):
                engine.validate_renderer_setup(config,{'use_corrections':True})


if __name__=='__main__':
    unittest.main()
