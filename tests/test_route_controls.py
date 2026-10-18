"""The cyclic control transfers complete candidate routes at offset eleven."""
from copy import deepcopy
import hmac
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tefed.data import Manifest,SampleRecord
from tefed import core
from tefed.diagnostics import renderer_target_observation
from tefed.server import build_cyclic_routes,construct_cases
from tests.test_server import (source_descriptor,services,config_for,calibration,
    FakeBank,FakeRouter,FakeRenderer,STATE)


class CyclicRouteControlTests(unittest.TestCase):
    def population(self):
        manifest=Manifest(tuple(SampleRecord(image,'client1','unlabeled',Path('never-open.png'))
                                for image in ('image','zz-image')))
        descriptor=source_descriptor();events=[]
        for index in range(12):
            image='image' if index<6 else 'zz-image'
            candidate_id=f'{index%6:02d}'
            route=[0,index%5+1,(index+1)%5+1]
            general=.45+index*.001;specialist=.32-index*.001
            event={'client_id':'client1','round_index':3,'candidate_id':candidate_id,
                'local_identity':hmac.new(b'secret',image.encode(),'sha256').hexdigest()[:32],
                'route':route,'weights':[general,specialist,.23],
                'disposition':'qualified' if index%2 else 'suppressed'}
            if index==0:
                event.update(descriptor_id=descriptor['id'],source_observation_key=descriptor['source'],disposition='correction')
            events.append(event)
        return manifest,descriptor,events

    def test_twelve_candidates_transfer_donor_tuple_and_keep_same_image_positions(self):
        manifest,d,events=self.population();before=deepcopy(events)
        mapping=build_cyclic_routes(events,manifest,b'secret',3,run_seed=8)
        reference=mapping[d['id']]
        self.assertEqual(reference['population_size'],12)
        self.assertEqual(reference['recipient_position'],0)
        self.assertEqual(reference['donor_position'],11)
        self.assertEqual(reference['donor_candidate_id'],'05')
        self.assertEqual(reference['donor_local_identity'],events[11]['local_identity'])
        self.assertEqual(reference['route'],events[11]['route'])
        self.assertEqual(reference['weights'],events[11]['weights'])
        self.assertEqual(events,before)
        self.assertNotIn('image_id',reference)
        self.assertEqual(len(reference['ordering_sha256']),64)

    def test_mapping_is_independent_of_upload_order_and_contains_wraparound(self):
        manifest,d,events=self.population()
        events[-1]['descriptor_id']='last-descriptor'
        expected=build_cyclic_routes(events,manifest,b'secret',3,run_seed=8)
        actual=build_cyclic_routes(list(reversed(events)),manifest,b'secret',3,run_seed=8)
        self.assertEqual(actual,expected)
        last=actual['last-descriptor']
        self.assertEqual(last['donor_position'],10)
        self.assertEqual(last['weights'],events[10]['weights'])

    def test_server_uses_full_tuple_preserves_source_and_exports_frozen_inputs(self):
        manifest,d,events=self.population()
        mapping=build_cyclic_routes(events,manifest,b'secret',3,run_seed=8)
        renderer=FakeRenderer();config=config_for(services(renderer),'cyclic')
        before=deepcopy(d)
        with TemporaryDirectory() as directory:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d['source_state_key']:STATE},
                ['client1'],calibration(),config,3,8,directory,cyclic_routes=mapping)
        case=result['cases'][0];reference=case['verification_reference']
        self.assertEqual(reference,mapping[d['id']])
        self.assertEqual(case['d']['route'],d['route'])
        self.assertEqual(case['d']['weights'],d['weights'])
        self.assertEqual(d,before)
        entry=next(row for row in result['audit'] if row.get('entered_verification'))
        frozen=entry['frozen_verification_inputs']
        self.assertEqual(frozen['descriptor']['route'],d['route'])
        self.assertEqual(frozen['cyclic_reference'],reference)
        self.assertEqual(frozen['current_reference']['route'],[0,3,4])
        self.assertEqual(frozen['target_class'],0)
        self.assertEqual(frozen['target_box'],[.2,.2,.6,.6])
        self.assertEqual(frozen['attrs_measured'],entry['fidelity_measurements'])
        self.assertEqual(set(frozen['expert_outputs']),set(map(str,range(6))))

    def test_stored_mode_exports_same_cyclic_reference_without_using_it(self):
        manifest,d,events=self.population()
        mapping=build_cyclic_routes(events,manifest,b'secret',3,run_seed=8)
        config=config_for(services(FakeRenderer()),'stored')
        with TemporaryDirectory() as directory:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d['source_state_key']:STATE},
                ['client1'],calibration(),config,3,8,directory,cyclic_routes=mapping)
        entry=next(row for row in result['audit'] if row.get('entered_verification'))
        self.assertEqual(entry['verification_reference']['route'],d['route'])
        self.assertEqual(entry['frozen_verification_inputs']['cyclic_reference'],mapping[d['id']])

    def test_cyclic_missing_or_malformed_mapping_fails_before_rendering(self):
        manifest,d,events=self.population()
        good=build_cyclic_routes(events,manifest,b'secret',3,run_seed=8)
        incomplete=deepcopy(good);incomplete[d['id']].pop('donor_candidate_id')
        wrong_round=deepcopy(good);wrong_round[d['id']]['round_index']=4
        for mapping in (None,{},incomplete,wrong_round):
            renderer=FakeRenderer();config=config_for(services(renderer),'cyclic')
            with self.subTest(mapping=mapping),TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError,'Cyclic'):
                    construct_cases([d],FakeBank(),FakeRouter(),{d['source_state_key']:STATE},
                        ['client1'],calibration(),config,3,8,directory,cyclic_routes=mapping)
                self.assertEqual(renderer.requests,[])

    def test_population_rejects_mixed_round_duplicate_or_unknown_observation(self):
        manifest,_,events=self.population()
        mixed=deepcopy(events);mixed[0]['round_index']=2
        unknown=deepcopy(events);unknown[0]['local_identity']='unknown'
        for bad in (mixed,unknown,events+[deepcopy(events[0])]):
            with self.assertRaises(ValueError):
                build_cyclic_routes(bad,manifest,b'secret',3,run_seed=8)

    def test_target_associated_outputs_preserve_verification_and_target_observation(self):
        d=source_descriptor();target={'box':[.2,.2,.6,.6],'class_id':0}
        outputs={str(head):[
            {'box':[.75,.75,.95,.95],'posterior':[.99,.01],'index':0},
            {'box':list(target['box']),'posterior':[.9,.1],'index':2},
            {'box':[.21,.21,.59,.59],'posterior':[.8,.2],'index':1}]
            for head in range(6)}
        outputs['5']=[]
        compact={}
        for head in range(6):
            matched=core.associate_hypothesis(outputs[str(head)],target['box'])
            compact[str(head)]=[matched] if matched else []
        variant={'id':'variant','target_box':target['box'],'attrs_measured':[0.]*8,'geometry':1.}
        for route,weights in (([0,1,2],[.5,.3,.2]),([0,3,4],[.45,.32,.23]),([0,4,5],[.45,.32,.23])):
            descriptor={**d,'route':route,'weights':weights}
            self.assertEqual(core.verify_case(variant,descriptor,outputs),core.verify_case(variant,descriptor,compact))
            self.assertEqual(renderer_target_observation(outputs,route,weights,target,scope='same'),
                             renderer_target_observation(compact,route,weights,target,scope='same'))

    def test_coordinator_passes_all_candidate_routes_to_stored_verification(self):
        import torch
        from tefed import engine
        from tefed.software_fixture import create_fixture
        torch.set_num_threads(2)
        original_assess=engine.assess_frame;captured=[]
        def correction_assessment(model,router,sample,*args,**kwargs):
            assessed=original_assess(model,router,sample,*args,**kwargs)
            if sample['split']=='unlabeled':
                for candidate in assessed['candidates']:
                    candidate.update(disposition='correction',attrs=[.5]*8,availability=[1]*8,rule_flags=[1,0,0,0,0])
            return assessed
        def capture(descriptors,*args,cyclic_routes=None,**kwargs):
            captured.append((deepcopy(descriptors),deepcopy(cyclic_routes)))
            return {'cases':[],'audit':[]}
        with TemporaryDirectory() as directory:
            root=Path(directory);config=create_fixture(root/'data')
            config['experiment']['rounds']=1
            config['renderer']['enabled']=True;config['decision']['render_budget']=1
            with patch('tefed.engine.validate_renderer_setup'),patch('tefed.engine.run_binding',return_value={'renderer_test':True}),\
                    patch('tefed.engine.assess_frame',side_effect=correction_assessment),\
                    patch('tefed.server.construct_cases',side_effect=capture):
                engine.run_experiment(config,seed=2,output_dir=root/'run')
        self.assertEqual(len(captured),1)
        descriptors,mapping=captured[0]
        self.assertEqual(set(mapping),{d['id'] for d in descriptors})
        self.assertGreater(len(mapping),1)
        self.assertTrue(all(row['run_seed']==2 and row['round_index']==0 for row in mapping.values()))


if __name__=='__main__':
    unittest.main()
