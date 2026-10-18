"""Per-case exposure is recorded at actual optimizer attempts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from tefed.engine import train_local
from tefed.model import RoutedYOLO


class CaseExposureTests(unittest.TestCase):
    def setup_training(self,device='cpu',beta=.5):
        torch.set_num_threads(2);torch.manual_seed(8);np.random.seed(8)
        config={'experiment':{'rounds':1,'local_epochs':2,'batch_size':2},
                'model':{'architecture':'yolo26n.yaml','classes':2,'specialists':0,'image_size':64},
                'optimizer':{'amp':device=='cuda','amp_init_scale':1.,'learning_rate':.001},
                'loss':{'beta':beta,'rho':0.,'gamma':0.}}
        model=RoutedYOLO(config).to(device)
        sample={'rgb':np.full((64,64,3),128,np.uint8),'depth':None,'nir':None,
                'targets':[{'box':[.2,.2,.7,.8],'class_id':0}]}
        items=[{'kind':'correction','sample':deepcopy(sample),
                'case':{'id':case_id,'d':{'route':[0],'weights':[1.]}}} for case_id in ('case-a','case-b')]
        return model,items,config

    def assert_counts(self,ledger):
        cases=ledger['case_presentations']
        self.assertEqual(sum(row['presentations'] for row in cases.values()),ledger['presentations']['correction'])
        self.assertEqual(ledger['presentations']['correction'],4)
        for row in cases.values():
            self.assertEqual(row['presentations'],2)
            self.assertEqual(row['attempted_step_ids'],[0,1])
            self.assertEqual(sorted(row['applied_step_ids']+row['amp_skipped_step_ids']),[0,1])
            self.assertEqual(len(row['applied_step_ids']),ledger['optimizer_steps'])
            self.assertEqual(len(row['amp_skipped_step_ids']),ledger['amp_skipped_steps'])

    def test_native_minibatch_case_counts_and_zero_based_steps(self):
        model,items,config=self.setup_training()
        ledger=train_local(model,items,config)
        self.assert_counts(ledger)
        self.assertTrue(all(row['supervised_presentations']==2 for row in ledger['case_presentations'].values()))
        self.assertTrue(all(row['detection_supervision_enabled'] for row in ledger['case_presentations'].values()))

    def test_beta_zero_preserves_presentations_without_detection_supervision(self):
        model,items,config=self.setup_training(beta=0.)
        ledger=train_local(model,items,config)
        self.assert_counts(ledger)
        self.assertTrue(all(row['supervised_presentations']==0 for row in ledger['case_presentations'].values()))
        self.assertTrue(all(not row['detection_supervision_enabled'] for row in ledger['case_presentations'].values()))

    def test_selected_event_links_descriptor_and_actual_beta_zero_exposure(self):
        from tefed.client import execute_client_round
        from tefed.data import load_manifest
        from tefed.model import Router
        from tefed.software_fixture import create_fixture
        from tests.test_core import STATE,descriptor
        with TemporaryDirectory() as directory:
            config=create_fixture(Path(directory))
            config['loss']['beta']=0.
            manifest=load_manifest(config['experiment']['manifest'])
            calibration=json.loads(Path(config['experiment']['calibration']).read_text())
            record=next(record for record in manifest.by_split('labeled','client0') if record.labels)
            d=descriptor()
            case={'id':'correction-case','round':0,'d':d,'rgb':str(record.rgb),
                  'targets':[label.as_dict() for label in record.labels]}
            result=execute_client_round(RoutedYOLO(config),Router(.5),manifest,STATE,calibration,config,
                client_id='client0',round_index=1,selected_cases=[case],run_secret=b's'*32)
            event=result['selected_events'][0]
            canonical=json.dumps({'client_id':'client0','round_index':1,'case_id':'correction-case'},sort_keys=True,separators=(',',':'))
            self.assertEqual(event['selection_event_id'],hashlib.sha256(canonical.encode()).hexdigest())
            self.assertEqual(event['descriptor_id'],d['id'])
            self.assertEqual(event['source_observation_key'],d['source'])
            self.assertEqual(event['source_route'],d['route'])
            self.assertEqual(event['actual_presentations'],1)
            self.assertEqual(event['supervised_presentations'],0)
            self.assertFalse(event['detection_supervision_enabled'])
            self.assertEqual(len(event['attempted_step_ids']),1)
            counts=result['ledger']['candidate_counts']
            self.assertEqual(sum(counts[name] for name in ('qualified','correction','suppressed')),counts['total'])

    @unittest.skipUnless(torch.cuda.is_available(),'Requires CUDA')
    def test_native_amp_overflow_links_cases_to_skipped_step(self):
        model,items,config=self.setup_training('cuda')
        calls=[]
        def overflow_once(grad):
            calls.append(True)
            return torch.full_like(grad,float('inf')) if len(calls)==1 else grad
        parameter=next(model.trunk.parameters())
        hook=parameter.register_hook(overflow_once)
        try:
            ledger=train_local(model,items,config)
        finally:
            hook.remove()
        self.assert_counts(ledger)
        self.assertEqual(ledger['amp_skipped_steps'],1)
        self.assertEqual(ledger['optimizer_steps'],1)
        for row in ledger['case_presentations'].values():
            self.assertEqual(row['amp_skipped_step_ids'],[0])
            self.assertEqual(row['applied_step_ids'],[1])


if __name__=='__main__':
    unittest.main()
