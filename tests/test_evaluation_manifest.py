"""Local and remote evaluation record the same frozen inference protocol."""
from copy import deepcopy
from dataclasses import replace
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from tefed import engine
from tefed.client import evaluate_client
from tefed.data import load_manifest
from tefed.evaluation_manifest import build_evaluation_manifest,identity_sha256
from tefed.model import RoutedYOLO,Router
from tefed.software_fixture import create_fixture
from tests.test_core import STATE


class EvaluationManifestTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(19)
        self.temporary=TemporaryDirectory();self.root=Path(self.temporary.name)
        self.config=create_fixture(self.root/'data')
        self.config['experiment']['seed']=3
        self.config['experiment']['device']='cuda' if torch.cuda.is_available() else 'cpu'
        self.manifest=load_manifest(self.config['experiment']['manifest'])
        records=[]
        for record in self.manifest.records:
            metadata={'rule_flags':['not-a-condition-tag']}
            if record.id==self.manifest.by_split('test')[0].id:
                metadata.update(paired_group_id='paired-frame-1',condition_tags=['night','rain'],condition_tags_version='conditions-v1',
                    condition_tag='night-rain',tag_definition_version='original-tag-v2')
            records.append(replace(record,metadata=metadata))
        self.manifest=replace(self.manifest,records=tuple(records),
            metadata={**self.manifest.metadata,'class_names':['a','b'],'label_dictionary_version':'labels-v1',
                      'condition_tags_version':'conditions-v1','tag_definition_version':'original-tag-v2'})
        self.calibration=json.loads(Path(self.config['experiment']['calibration']).read_text())
        self.states={client:deepcopy(STATE) for client in ('client0','client1')}
        self.model=RoutedYOLO(self.config).to(self.config['experiment']['device'])
        self.router=Router(.5).to(self.config['experiment']['device'])

    def tearDown(self):
        self.temporary.cleanup()

    def local_evaluation(self):
        directory=self.root/'local'
        metrics=engine.evaluate_model(self.model,self.router,self.manifest,self.states,self.calibration,self.config,directory)
        return metrics,json.loads((directory/'evaluation_manifest.json').read_text())

    def remote_executor(self,*,tamper=False):
        outer=self
        class Executor:
            def execute(self,action,model,router,calibration,*,client_id,state,persistent_state):
                result=evaluate_client(deepcopy(model),deepcopy(router),outer.manifest,state,calibration,outer.config,
                    client_id=client_id,persistent_state=persistent_state)
                if tamper:
                    if tamper=='geometry':
                        result['views']['unseen']['audit_records'][0]['letterbox']['pad']=[1,1]
                    else:
                        result['views']['unseen']['audit_records'][0]['transform_sha256']='0'*64
                return result
        return Executor()

    def test_native_local_and_remote_manifests_match_without_coordinator_images(self):
        metrics,local=self.local_evaluation()
        coordinator=replace(self.manifest,records=tuple(replace(record,rgb=self.root/'absent'/f'{record.id}.png',depth=None,nir=None)
                                                       for record in self.manifest.records))
        remote_metrics,_=engine.evaluate_remote_models(self.model,self.router,coordinator,self.states,self.calibration,self.config,
            self.root/'remote',self.remote_executor())
        remote=json.loads((self.root/'remote'/'predictions'/'evaluation_manifest.json').read_text())
        self.assertEqual(metrics,remote_metrics)
        self.assertEqual(local['views'],remote['views'])
        self.assertEqual(local['model_identity'],remote['model_identity'])
        self.assertEqual(local['schema_version'],1)
        self.assertEqual(local['seed'],3)
        self.assertEqual(local['label_dictionary']['version'],'labels-v1')
        self.assertEqual(local['model_identity']['kind'],'state_dict_sha256')
        self.assertNotIn('checkpoint_sha256',local['model_identity'])
        self.assertEqual(local['frozen_states_sha256'],identity_sha256(self.states))
        self.assertEqual(local['calibration_sha256'],identity_sha256(self.calibration))
        for view,record in local['prediction_files'].items():
            path=self.root/'local'/f'{view}.jsonl'
            self.assertEqual(record['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record['bytes'],path.stat().st_size)
            self.assertEqual(record['path'],path.name)
        records=local['views']['participating']['records']
        tagged=next(row for row in records if row['paired_group_id'])
        self.assertEqual(tagged['condition_tags'],['night','rain'])
        self.assertEqual(tagged['condition_tag'],'night-rain')
        self.assertEqual(tagged['tag_definition_version'],'original-tag-v2')
        self.assertEqual(local['tag_definition_version'],'original-tag-v2')
        self.assertTrue(any(row['condition_tags'] is None for row in records))
        self.assertTrue(all(row['state_source']=='calibration_prior' for row in local['views']['new-client']['records']))

    def test_remote_rejects_transform_protocol_mismatch(self):
        with self.assertRaisesRegex(ValueError,'transform protocol'):
            engine.evaluate_remote_models(self.model,self.router,self.manifest,self.states,self.calibration,self.config,
                self.root/'mismatch',self.remote_executor(tamper=True))
        self.assertFalse((self.root/'mismatch'/'predictions'/'evaluation_manifest.json').exists())

    def test_remote_rejects_letterbox_geometry_mismatch(self):
        with self.assertRaisesRegex(ValueError,'letterbox geometry'):
            engine.evaluate_remote_models(self.model,self.router,self.manifest,self.states,self.calibration,self.config,
                self.root/'geometry-mismatch',self.remote_executor(tamper='geometry'))

    def test_undeclared_versions_and_tags_remain_null(self):
        self.manifest=replace(self.manifest,records=tuple(replace(record,metadata={'rule_flags':[True]*8})
                                                        for record in self.manifest.records),
            metadata={key:value for key,value in self.manifest.metadata.items()
                      if key not in {'label_dictionary_version','condition_tags_version'}})
        _,local=self.local_evaluation()
        self.assertIsNone(local['label_dictionary']['version'])
        self.assertIsNone(local['condition_tags_version'])
        for data in local['views'].values():
            self.assertTrue(all(row['condition_tags'] is None and row['paired_group_id'] is None for row in data['records']))

    def test_bundled_model_and_bn_identities_are_explicit(self):
        _,local=self.local_evaluation()
        records={view:data['records'] for view,data in local['views'].items()}
        models={client:deepcopy(self.model.state_dict()) for client in self.states}
        bundled=build_evaluation_manifest(self.model,self.router,self.manifest,self.states,self.calibration,self.config,records,
            client_models=models,local_new_client_id='client1')
        self.assertEqual(bundled['model_identity']['client_models_sha256'],identity_sha256(models))
        self.assertTrue(all(row['model_selector']['client_id']=='client1' for row in bundled['views']['new-client']['records']))
        key=next(iter(self.model.state_dict()))
        bn={'client0':{key:self.model.state_dict()[key]}}
        normalized=build_evaluation_manifest(self.model,self.router,self.manifest,self.states,self.calibration,self.config,records,client_bn=bn)
        self.assertEqual(normalized['model_identity']['client_bn_sha256'],identity_sha256(bn))

    def test_manifest_rejects_missing_declared_identity(self):
        _,local=self.local_evaluation()
        records={view:data['records'] for view,data in local['views'].items()}
        records['hard']=records['hard'][:-1]
        with self.assertRaisesRegex(ValueError,'audit identities'):
            build_evaluation_manifest(self.model,self.router,self.manifest,self.states,self.calibration,self.config,records)


if __name__=='__main__':
    unittest.main()
