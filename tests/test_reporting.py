"""Measured report inputs, identity joins, and fixed qualitative geometry."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from tefed import core
from tefed.data import load_manifest, load_sample
from tefed.evaluation_manifest import identity_sha256, record_metadata
from tefed.reporting import audit_run, export_audit, plot_runs, qualitative_comparison


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf-8')


def make_run(path,seed=0,method='tefed'):
    path.mkdir()
    config={'experiment':{'method':method,'seed':seed,'rounds':2},'model':{'image_size':64,'classes':2},
            'evaluation':{'views':['hard']}}
    write(path/'configuration.json',config)
    write(path/'environment.json',{'seed':seed,'manifest_sha256':'a'*64,'calibration_sha256':'b'*64,
        'initial_checkpoint_sha256':'c'*64,'versions':{'fixture':'1'}})
    write(path/'status.json',{'status':'completed','completed_rounds':2})
    write(path/'metrics.json',{'hard':{'mAP50':.2+seed*.1,'AP75':.1,'mAP50_95':.15}})
    candidate={'disposition':'correction','id':'anchor1','candidate_id':'anchor1','class_id':0,
        'route':[0,1,2],'weights':[.5,.3,.2]}
    descriptor=core.make_descriptor(candidate,[.5]*8,[1]*8,[1]*5,'client1','sample1',0,
        {'q':.8,'p':.7,'h':[.2]*5},'secret')
    event={**candidate,'round_index':0,'client_id':'client1','local_identity':'private-key',
        'descriptor_id':descriptor['id'],'source_observation_key':descriptor['source']}
    jsonl(path/'client_key_candidate_audit.jsonl',[event])
    case_id='case1'; request={'case_id':case_id,'descriptor':descriptor}
    write(path/'rendering'/'round_000'/case_id/'request.json',request)
    outputs={str(head):[{'box':[.1,.1,.3,.3],'posterior':[.95,.05],'index':0}] for head in range(6)}
    verification=core.verify_case({'id':case_id,'target_box':[.1,.1,.3,.3],'attrs_measured':[.5]*8,'geometry':1.},descriptor,outputs)
    assert verification['retained']
    record={'case_id':case_id,'descriptor_id':descriptor['id'],'creation_round':0,'rendered_variant_id':case_id,
        'source_observation_key':descriptor['source'],'source_client_id':'client1','source_state_key':descriptor['source_state_key'],
        'source_route':descriptor['route'],'source_weights':descriptor['weights'],'entered_verification':True,**verification}
    write(path/'verification_round_000.json',[record]); write(path/'verification_round_001.json',[])
    write(path/'final.pool.json',[{'id':case_id,'round':0,'d':descriptor,'fidelity':verification['fidelity']}])
    tracked={'presentations':2,'supervised_presentations':2,'detection_supervision_enabled':True,
        'attempted_step_ids':[0,1],'applied_step_ids':[0],'amp_skipped_step_ids':[1]}
    ledgers=[]
    for round_index in (0,1):
        ledger={'optimizer_steps':1,'attempted_optimizer_steps':1 if round_index==0 else 2,'amp_skipped_steps':round_index,
            'presentations':{'correction':round_index*2},'case_presentations':{} if round_index==0 else {case_id:tracked},
            'candidate_counts':{'qualified':0,'correction':1 if round_index==0 else 0,'suppressed':0,'total':1 if round_index==0 else 0},
            'mean_step_loss':1.-round_index*.1}
        ledgers.append({'round_index':round_index,'clients':{'client1':ledger},
            'reuse_selected_ids':{'client1':[] if round_index==0 else [case_id]}})
    jsonl(path/'rounds.jsonl',ledgers)
    identity={'client_id':'client1','round_index':1,'case_id':case_id}
    selection_id=sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    selected={**identity,'selection_event_id':selection_id,'descriptor_id':descriptor['id'],
        'source_observation_key':descriptor['source'],'pattern_key':descriptor['pattern'],'creation_round':0,
        'source_route':descriptor['route'],'source_weights':descriptor['weights'],'actual_presentations':2,**tracked}
    jsonl(path/'selected_events.jsonl',[selected])
    return path


def make_qualitative(root,view='hard',*,sidecar=False):
    root.mkdir()
    Image.new('RGB',(80,40),(120,60,30)).save(root/'rgb.png')
    Image.new('L',(80,40),80).save(root/'nir.png')
    manifest_path=root/'manifest.json'
    write(manifest_path,{'records':[{'id':'sample','client_id':'client1','split':'hard' if view=='hard' else 'test',
        'rgb':'rgb.png','nir':'nir.png','labels':[{'box':[.1,.1,.3,.4],'class_id':0}]}],
        'views':{view:['sample']},'class_names':['target']})
    manifest=load_manifest(manifest_path)
    sample=load_sample(manifest.by_id['sample'],size=64)
    geometry_hash=sha256(json.dumps(sample['meta'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    paths={}
    for label in ('Method A','Method B'):
        path=root/label/'hard.jsonl'
        row={'id':'sample','view':view,'coordinate_system':'letterbox-normalized-xyxy','image_size':64,
            'transform_hash_kind':'sample-metadata-sha256-v1','transform_sha256':geometry_hash,
            'predictions':[{'box':sample['targets'][0]['box'],'class_id':0,'score':.7}]}
        jsonl(path,[row]); paths[label]=path
        if sidecar:
            protocol={'image_size':64,'seed':0,'unseen':None}
            protocol_hash=sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            evidence={'schema_version':1,'coordinate_system':'letterbox-normalized-xyxy',
                'model_identity':{'kind':'state_dict_sha256','sha256':'d'*64},
                'manifest_sha256':identity_sha256(manifest.as_dict()),
                'prediction_files':{view:{'sha256':sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}},
                'views':{view:{'transform':protocol,'records':[{**record_metadata(manifest.by_id['sample']),'letterbox':sample['meta'],
                    'applied_unseen':None,'transform_sha256':protocol_hash}]}}}
            write(path.parent/'evaluation_manifest.json',evidence)
    return manifest_path,paths,sample


class RunAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.run=make_run(self.root/'run')

    def test_nonempty_linked_case_exact_exposure_and_export(self):
        report,_=audit_run(self.run)
        self.assertEqual(report['status'],'passed',report)
        self.assertEqual(report['exposure'][1]['actual_presentations'],2)
        out=self.root/'export'; result=export_audit(self.run,out)
        self.assertEqual(result['status'],'passed')
        for name in ('audit.json','cases.csv','selections.csv','fidelity.csv','source_manifest.json'):
            self.assertTrue((out/name).is_file())
        self.assertFalse(json.loads((out/'source_manifest.json').read_text())['rules'].get('imputation',False))
        with self.assertRaises(FileExistsError):
            export_audit(self.run,out)

    def test_incomplete_round_sequence_rejected(self):
        rows=[json.loads(line) for line in (self.run/'rounds.jsonl').read_text().splitlines()]
        jsonl(self.run/'rounds.jsonl',rows[1:])
        with self.assertRaisesRegex(ValueError,'sequence'):
            audit_run(self.run)

    def test_boolean_committed_round_rejected(self):
        write(self.run/'status.json',{'status':'completed','completed_rounds':True})
        with self.assertRaisesRegex(ValueError,'status'):
            audit_run(self.run)

    def test_old_missing_evidence_is_incomplete(self):
        rows=[json.loads(line) for line in (self.run/'rounds.jsonl').read_text().splitlines()]
        for row in rows:
            del row['clients']['client1']['candidate_counts']
        jsonl(self.run/'rounds.jsonl',rows)
        report,_=audit_run(self.run)
        self.assertEqual(report['status'],'incomplete')
        self.assertTrue(any('candidate_counts' in entry for entry in report['missing']))

    def test_duplicate_candidate_rejected_even_without_descriptor(self):
        path=self.run/'client_key_candidate_audit.jsonl'; row=json.loads(path.read_text())
        row.pop('descriptor_id'); jsonl(path,[row,row])
        report,_=audit_run(self.run)
        self.assertEqual(report['status'],'invalid')
        self.assertTrue(any('duplicate candidate' in entry for entry in report['errors']))

    def test_source_route_tampering_and_fixed_fidelity_rule_rejected(self):
        path=self.run/'rendering'/'round_000'/'case1'/'request.json'; request=json.loads(path.read_text())
        request['descriptor']['weights']=[.4,.3,.3]; write(path,request)
        report,_=audit_run(self.run)
        self.assertTrue(any('differs from uploaded source' in entry for entry in report['errors']))
        path=self.run/'verification_round_000.json'; rows=json.loads(path.read_text())
        rows[0]['fidelity_audit']['weights'][0]=.2; write(path,rows)
        report,_=audit_run(self.run)
        self.assertTrue(any('fixed fidelity' in entry for entry in report['errors']))

    def test_same_round_and_event_hash_tampering_rejected(self):
        path=self.run/'final.pool.json'; rows=json.loads(path.read_text()); rows[0]['round']=1; write(path,rows)
        path=self.run/'selected_events.jsonl'; row=json.loads(path.read_text()); row['selection_event_id']='x'; jsonl(path,[row])
        report,_=audit_run(self.run)
        self.assertTrue(any('same/future' in entry for entry in report['errors']))
        self.assertTrue(any('identity hash' in entry for entry in report['errors']))

    def test_case_steps_must_partition_outcomes(self):
        path=self.run/'rounds.jsonl'; rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['clients']['client1']['case_presentations']['case1']['amp_skipped_step_ids']=[0]
        jsonl(path,rows); report,_=audit_run(self.run)
        self.assertTrue(any('partition' in entry for entry in report['errors']))

    def test_selection_exposure_must_equal_training(self):
        path=self.run/'selected_events.jsonl'; row=json.loads(path.read_text()); row['actual_presentations']=3; jsonl(path,[row])
        report,_=audit_run(self.run)
        self.assertTrue(any('actual presentations differ' in entry for entry in report['errors']))

    def test_shared_case_step_outcome_conflict_rejected(self):
        pool=json.loads((self.run/'final.pool.json').read_text()); second=deepcopy(pool[0]); second['id']='case2'; pool.append(second); write(self.run/'final.pool.json',pool)
        request=json.loads((self.run/'rendering'/'round_000'/'case1'/'request.json').read_text()); request['case_id']='case2'
        write(self.run/'rendering'/'round_000'/'case2'/'request.json',request)
        path=self.run/'verification_round_000.json'; verification=json.loads(path.read_text()); record=deepcopy(verification[0]); record.update(case_id='case2',rendered_variant_id='case2'); verification.append(record); write(path,verification)
        path=self.run/'selected_events.jsonl'; events=[json.loads(line) for line in path.read_text().splitlines()]; event=deepcopy(events[0]); event.update(case_id='case2',attempted_step_ids=[0],applied_step_ids=[],amp_skipped_step_ids=[0])
        event['selection_event_id']=sha256(json.dumps({'client_id':'client1','round_index':1,'case_id':'case2'},sort_keys=True,separators=(',',':')).encode()).hexdigest(); events.append(event); jsonl(path,events)
        path=self.run/'rounds.jsonl'; rounds=[json.loads(line) for line in path.read_text().splitlines()]
        rounds[1]['reuse_selected_ids']['client1'].append('case2'); ledger=rounds[1]['clients']['client1']; ledger['presentations']['correction']=4
        tracked=deepcopy(ledger['case_presentations']['case1']); tracked.update(attempted_step_ids=[0],applied_step_ids=[],amp_skipped_step_ids=[0]); ledger['case_presentations']['case2']=tracked; jsonl(path,rounds)
        report,_=audit_run(self.run)
        self.assertTrue(any('conflicting case outcomes' in entry for entry in report['errors']))

    def test_unevaluated_fidelity_is_null_and_not_counted_as_pass(self):
        path=self.run/'verification_round_000.json'; rows=json.loads(path.read_text()); record=rows[0]
        descriptor=json.loads((self.run/'final.pool.json').read_text())[0]['d']
        record.update(retained=False,fidelity=None,reason='coverage',first_failed_gate='coverage',
            fidelity_audit=core.fidelity_audit(descriptor,first_failed_gate='coverage'))
        write(path,rows); report,_=audit_run(self.run)
        self.assertIsNone(report['fidelity_rows'][0]['score'])
        self.assertFalse(report['fidelity_rows'][0]['evaluated'])
        self.assertEqual(report['status'],'invalid')


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name); self.addCleanup(self.temp.cleanup)

    def test_measured_curves_and_seed_sd(self):
        runs=[make_run(self.root/f'run{seed}',seed=seed) for seed in (2,4)]
        result=plot_runs(runs,self.root/'plots',seeds=[2,4])
        self.assertEqual(result['series'],2)
        values=(self.root/'plots'/'plot_values.csv').read_text()
        self.assertIn('0.6000000000000001',values)
        self.assertTrue((self.root/'plots'/'training_curves.svg').is_file())
        with self.assertRaises(FileExistsError):
            plot_runs(runs,self.root/'plots',seeds=[2,4])

    def test_different_evaluation_settings_rejected(self):
        left=make_run(self.root/'a'); right=make_run(self.root/'b',method='fedavg')
        path=right/'configuration.json'; config=json.loads(path.read_text()); config['evaluation']['unseen_brightness']=.7; write(path,config)
        with self.assertRaisesRegex(ValueError,'share data'):
            plot_runs([left,right],self.root/'plots')

    def test_wrong_seed_label_rejected(self):
        run=make_run(self.root/'a'); environment=json.loads((run/'environment.json').read_text()); environment['seed']=3; write(run/'environment.json',environment)
        with self.assertRaisesRegex(ValueError,'seed differs'):
            plot_runs([run],self.root/'plots')

    def test_single_seed_has_no_invented_uncertainty(self):
        run=make_run(self.root/'a'); plot_runs([run],self.root/'plots')
        self.assertIn('SD unavailable',(self.root/'plots'/'metric_comparison.svg').read_text())

    def test_factorial_actual_complete_cells_and_declared_seeds(self):
        run=make_run(self.root/'a'); records=[]; jobs=[]; entries=[]; index=[]
        for seed in (2,4):
            for a in (0,1):
                for b in (0,1):
                    for c in (0,1):
                        factors={'A':a,'B':b,'C':c}; variant=f'{a}{b}{c}'
                        output=make_run(self.root/f'factor_{seed}_{variant}',seed=seed)
                        config=json.loads((output/'configuration.json').read_text()); config['factorial']={'factors':factors}; write(output/'configuration.json',config)
                        metrics=json.loads((output/'metrics.json').read_text())
                        job={'seed':seed,'factors':factors,'output':str(output),'variant':variant}; jobs.append(job)
                        entries.append({**job,'status':'completed','result':{'metrics':metrics}})
                        index.append({**job,'status':'completed','metrics':metrics,'protocol_sha256':'e'*64})
                        records.append({'seed':seed,'factors':factors,'value':metrics['hard']['mAP50'],
                            'protocol_sha256':'e'*64})
        path=self.root/'factorial_hard_mAP50.jsonl'; jsonl(path,records)
        write(self.root/'plan.json',{'suite_plan':'factorial','suite_sha256':'s','protocol_sha256':'e'*64,'jobs':jobs})
        write(self.root/'execution.json',{'status':'completed','suite_sha256':'s','jobs':entries})
        jsonl(self.root/'factorial_runs.jsonl',index)
        with self.assertRaisesRegex(ValueError,'Missing declared seed'):
            plot_runs([run],self.root/'plots',factorial_records=path,seeds=[2,4])
        runs=[make_run(self.root/f'seed{seed}',seed=seed) for seed in (2,4)]
        plot_runs(runs,self.root/'complete',factorial_records=path,seeds=[2,4])
        self.assertTrue((self.root/'complete'/'factorial_effects.svg').is_file())
        records[0]['value']=.99; jsonl(path,records)
        with self.assertRaisesRegex(ValueError,'differ from completed'):
            plot_runs(runs,self.root/'tampered',factorial_records=path,seeds=[2,4])


class QualitativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name); self.addCleanup(self.temp.cleanup)

    def test_fixed_letterbox_and_common_threshold_export(self):
        manifest,paths,sample=make_qualitative(self.root/'inputs')
        out=self.root/'qual'; result=qualitative_comparison(manifest,'hard',['sample'],paths,out,image_size=64,threshold=.75)
        self.assertEqual(result['ids'],['sample'])
        with Image.open(out/'qualitative.png') as image:
            self.assertEqual(image.size,(5*64,98))
        provenance=json.loads((out/'selection_manifest.json').read_text())
        self.assertEqual(provenance['confidence_threshold'],.75)
        self.assertEqual(provenance['records'][0]['targets'],sample['targets'])
        self.assertIsNone(provenance['records'][0]['condition_tag'])

    def test_json_roundtrip_evaluation_sidecar_geometry(self):
        manifest,paths,_=make_qualitative(self.root/'inputs',sidecar=True)
        qualitative_comparison(manifest,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)
        self.assertTrue((self.root/'qual'/'qualitative.png').is_file())

    def test_prediction_byte_change_rejected(self):
        manifest,paths,_=make_qualitative(self.root/'inputs',sidecar=True)
        path=paths['Method A']; row=json.loads(path.read_text()); row['predictions'][0]['score']=.6; jsonl(path,[row])
        with self.assertRaisesRegex(ValueError,'file identity'):
            qualitative_comparison(manifest,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)

    def test_different_dataset_sidecar_rejected(self):
        manifest,paths,_=make_qualitative(self.root/'inputs',sidecar=True)
        payload=json.loads(manifest.read_text()); payload['class_names']=['different']; write(manifest,payload)
        with self.assertRaisesRegex(ValueError,'dataset identity'):
            qualitative_comparison(manifest,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)

    def test_real_evaluation_manifest_json_roundtrip(self):
        from tefed.evaluation_manifest import build_evaluation_manifest, evaluation_record
        import numpy as np
        manifest_path,paths,sample=make_qualitative(self.root/'inputs')
        manifest=load_manifest(manifest_path)
        class Module:
            def state_dict(self):
                return {'w':np.ones((1,),dtype=np.float32)}
        config={'experiment':{'method':'tefed','seed':0},'model':{'image_size':64,'classes':1},'evaluation':{}}
        record=evaluation_record(manifest.by_id['sample'],sample,config,'hard',state_source='frozen_client_states')
        evidence=build_evaluation_manifest(Module(),Module(),manifest,{}, {'state_prior':{'q':.5,'p':.5,'h':[.2]*5}},config,{'hard':[record]})
        for path in paths.values():
            bound=deepcopy(evidence); bound['prediction_files']={'hard':{'sha256':sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}}
            write(path.parent/'evaluation_manifest.json',bound)
        qualitative_comparison(manifest_path,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)
        self.assertTrue((self.root/'qual'/'qualitative.png').is_file())

    def test_unbound_prediction_geometry_rejected(self):
        manifest,paths,_=make_qualitative(self.root/'inputs')
        path=paths['Method A']; row=json.loads(path.read_text()); row.pop('image_size'); jsonl(path,[row])
        with self.assertRaisesRegex(ValueError,'transform identity required'):
            qualitative_comparison(manifest,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)

    def test_different_coordinates_or_duplicate_ids_rejected(self):
        manifest,paths,_=make_qualitative(self.root/'inputs')
        path=paths['Method A']; row=json.loads(path.read_text()); row['coordinate_system']='original'; jsonl(path,[row])
        with self.assertRaisesRegex(ValueError,'coordinate'):
            qualitative_comparison(manifest,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)
        row['coordinate_system']='letterbox-normalized-xyxy'; jsonl(path,[row,row])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            qualitative_comparison(manifest,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)

    def test_unseen_requires_explicit_transformation_and_panel_labels(self):
        manifest,paths,_=make_qualitative(self.root/'inputs',view='unseen')
        with self.assertRaisesRegex(ValueError,'explicit perturbation'):
            qualitative_comparison(manifest,'unseen',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)
        methods={'rgb':paths['Method A'],'B':paths['Method B']}
        with self.assertRaisesRegex(ValueError,'panel names'):
            qualitative_comparison(manifest,'unseen',['sample'],methods,self.root/'qual',image_size=64,threshold=.5)

    def test_unmatched_view_membership_rejected(self):
        manifest,paths,_=make_qualitative(self.root/'inputs')
        path=paths['Method A']; row=json.loads(path.read_text()); row['id']='different'; jsonl(path,[row])
        with self.assertRaisesRegex(ValueError,'exactly match'):
            qualitative_comparison(manifest,'hard',['sample'],paths,self.root/'qual',image_size=64,threshold=.5)

    def test_cli_direct_help(self):
        result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts'/'report_runs.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('qualitative',result.stdout)


if __name__=='__main__':
    unittest.main()
