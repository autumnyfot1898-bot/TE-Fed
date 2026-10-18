"""Server coordination with fake inference and rendering services."""
from copy import deepcopy
import importlib.util
import hmac
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from tefed.core import make_descriptor
from tefed.server import construct_cases, router_objective, update_router, _load_case

HAS_TORCH=importlib.util.find_spec("torch") is not None

STATE={"q":.8,"p":.7,"h":[.2]*5}
PROTOTYPES=np.vstack([np.ones(5)/np.sqrt(5),np.eye(5)])


class FakeParameter:
    def __init__(self):
        self.requires_grad=True
    def requires_grad_(self,value):
        self.requires_grad=value


class FakeBank:
    def __init__(self):
        self.value=7
        self.training=True
        self.parameter=FakeParameter()
    def eval(self):
        self.training=False
        return self
    def parameters(self):
        return [self.parameter]


class FakeRouter(FakeBank):
    def __init__(self):
        super().__init__()
        self.prototypes=PROTOTYPES.copy()


def prediction(p=(.9,.1)):
    return {"box":[.2,.2,.6,.6],"posterior":list(p),"class_id":int(np.argmax(p)),"index":0}


def source_descriptor():
    candidate={"id":"source","class_id":0,"disposition":"correction","route":[0,1,2],"weights":[.5,.3,.2]}
    return make_descriptor(candidate,[0.]*8,[1]*8,[1,0,0,0,0],"client1","image",0,STATE,"secret")


class FakeRenderer:
    def __init__(self):
        self.requests=[]
    def render(self,request,output_dir):
        self.requests.append(deepcopy(request))
        return {"id":request["case_id"],"rgb":np.full((64,64,3),128,dtype=np.uint8),
                "depth":None,"nir":None,"enabled_modalities":["RGB"],
                "targets":[{"box":[.2,.2,.6,.6],"class_id":0}],"target_box":[.2,.2,.6,.6],
                "focal_target_index":0,"attrs_measured":[0.]*8,"metadata":{"test_backend":True}}


def services(renderer,posterior=(.9,.1),anchor=True,record=None):
    def assess(model,router,sample,state,calibration,config,round_id):
        if record is not None:
            record.append((model.value,model.parameter.requires_grad,model.training))
        p=prediction(posterior)
        candidate={**p,"id":"anchor","route":[0,3,4],"weights":[.5,.3,.2],"routing_vector":[.1]*17}
        return {"candidates":[candidate] if anchor else [],"vectors":[[.1]*17] if anchor else [],
                "modalities":{"RGB":[p]},"references":[p]}
    def predict(model,sample,calibration,config):
        return {"outputs":{str(m):[prediction(posterior)] for m in range(6)},"proposal_count":1}
    return {"renderer":renderer,"assess_frame":assess,"predict_outputs":predict,
            "load_case":lambda case,config:deepcopy(case)}


def config_for(service,mode="stored"):
    return {"experiment":{"method":"tefed"},"model":{"classes":2,"image_size":64},
            "decision":{"render_budget":1,"verification_route":mode},"renderer":{"enabled":True},
            "_server_services":service}


def calibration():
    return {"T_C":1.,"luminance_median":128/255,
            "class_priors":{"0":{"r_c_med":1.,"a_c":None,"b_c":None},"1":{"r_c_med":1.,"a_c":None,"b_c":None}},
            "loss_lookup":{"base":{name:[.1]*4 for name in ("illum","occ","ctx","mat","sens")}}}


class ServerConstructionTests(unittest.TestCase):
    def test_real_gates_and_round_start_snapshot_without_mutation(self):
        d=source_descriptor();before=deepcopy(d)
        renderer=FakeRenderer();observed=[];bank=FakeBank()
        cfg=config_for(services(renderer,record=observed))
        with TemporaryDirectory() as folder:
            result=construct_cases([d],bank,FakeRouter(),{d["source_state_key"]:STATE},["client1"],calibration(),cfg,3,8,folder)
        self.assertEqual(len(result["cases"]),1)
        case=result["cases"][0]
        self.assertEqual(case["round"],3)
        self.assertEqual(case["d"]["route"],[0,1,2])
        self.assertTrue(case["verification"]["retained"])
        self.assertEqual(observed,[(7,False,False)])
        self.assertTrue(bank.parameter.requires_grad)
        self.assertTrue(bank.training)
        self.assertEqual(d,before)
        self.assertEqual(len(renderer.requests),1)

    def test_bad_posterior_fails_margin_not_keep_all(self):
        d=source_descriptor();renderer=FakeRenderer()
        cfg=config_for(services(renderer,posterior=(.1,.9)))
        with TemporaryDirectory() as folder:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},["client1"],calibration(),cfg,0,0,folder)
        self.assertEqual(result["cases"],[])
        verification=[a for a in result["audit"] if "margin" in a]
        self.assertEqual(verification[0]["reason"],"margin")
        self.assertLess(verification[0]["margin"],0.)

    def test_current_and_cyclic_routes_do_not_overwrite_source(self):
        d=source_descriptor()
        for mode,route in (("current",[0,3,4]),("cyclic",[0,2,3])):
            renderer=FakeRenderer();cfg=config_for(services(renderer),mode)
            cyclic_routes=None
            if mode=='cyclic':
                from tefed.data import Manifest,SampleRecord
                from tefed.server import build_cyclic_routes
                manifest=Manifest((SampleRecord('image','client1','unlabeled',Path('unused.png')),))
                identity=hmac.new(b'secret',b'image','sha256').hexdigest()[:32]
                events=[{'client_id':'client1','local_identity':identity,'round_index':0,
                    'candidate_id':f'{i:02d}','route':[0,2,3],'weights':[.6,.25,.15],
                    **({'descriptor_id':d['id']} if i==0 else {})} for i in range(12)]
                cyclic_routes=build_cyclic_routes(events,manifest,b'secret',0,run_seed=0)
            with TemporaryDirectory() as folder:
                result=construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},["client1"],calibration(),cfg,0,0,folder,
                    cyclic_routes=cyclic_routes)
            case=result["cases"][0]
            self.assertEqual(case["verification_reference"]["route"],route)
            self.assertEqual(case["d"]["route"],[0,1,2])
            self.assertEqual(d["route"],[0,1,2])
            if mode=='cyclic':
                self.assertEqual(case['verification_reference']['weights'],[.6,.25,.15])
                self.assertEqual(case['verification_reference']['donor_candidate_id'],'11')

    def test_missing_current_anchor_and_unresolved_state_are_exclusions(self):
        d=source_descriptor();renderer=FakeRenderer()
        cfg=config_for(services(renderer,anchor=False),"current")
        with TemporaryDirectory() as folder:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},["client1"],calibration(),cfg,0,0,folder)
            missing=construct_cases([d],FakeBank(),FakeRouter(),{},["client1"],calibration(),cfg,0,0,folder)
        self.assertEqual(result["cases"],[])
        self.assertTrue(any(a["reason"]=="current_route_missing_general_anchor" for a in result["audit"]))
        self.assertEqual(missing["audit"][0]["reason"],"descriptor_rejected")

    def test_active_missing_fidelity_rejects_instead_of_copying_source(self):
        d=source_descriptor()
        d["attrs"][5]=.5
        d["mask"][5]=1
        from tefed.core import attribute_bins
        d["pattern"]=d["branch"]=sum(4**i*b for i,b in enumerate(attribute_bins(d["attrs"])))
        renderer=FakeRenderer();cfg=config_for(services(renderer))
        # Missing geometry under an active source mask must fail fidelity.
        original=cfg["_server_services"]["assess_frame"]
        def no_refs(*args,**kwargs):
            value=original(*args,**kwargs);value["references"]=None;return value
        cfg["_server_services"]["assess_frame"]=no_refs
        with TemporaryDirectory() as folder:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},["client1"],calibration(),cfg,0,0,folder)
        self.assertFalse(result["cases"])
        self.assertTrue(any(a.get("reason")=="undefined_active_measurement" for a in result["audit"]))

    def test_generic_prior_is_calibration_artifact_and_source_preserved(self):
        d=source_descriptor();renderer=FakeRenderer();cfg=config_for(services(renderer))
        cfg["experiment"]["method"]="generic_hc"
        cal=calibration();cal["generic_priors"]={"illum":{"attrs":[0.]*8,"availability":[1]*8}}
        with TemporaryDirectory() as folder:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},["client1"],cal,cfg,0,0,folder)
        self.assertEqual(result["cases"][0]["d"]["rule_flags"],[1,0,0,0,0])
        self.assertEqual(result["cases"][0]["original_source_descriptor"]["attrs"],d["attrs"])
        self.assertEqual(result["cases"][0]["verification_reference"]["route"],[0,3,4])

    def test_zero_render_budget_still_scores_upload_descriptors(self):
        d=source_descriptor();renderer=FakeRenderer();cfg=config_for(services(renderer))
        cfg["decision"]["render_budget"]=0
        with TemporaryDirectory() as folder:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},["client1"],calibration(),cfg,0,0,folder)
        self.assertEqual(result["cases"],[])
        self.assertEqual(len(renderer.requests),0)
        self.assertGreater(result["scored_descriptors"][0]["priority"],0.)

    def test_scene_class_mapping_passed_without_mutating_configuration(self):
        d=source_descriptor();renderer=FakeRenderer();cfg=config_for(services(renderer))
        mapping={"vehicle_body":2,"truck_body":3}
        cfg["renderer"]["scene_class_map"]=deepcopy(mapping)
        with TemporaryDirectory() as folder:
            construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},["client1"],calibration(),cfg,0,0,folder)
        self.assertEqual(renderer.requests[0]["scene_class_map"],mapping)
        self.assertEqual(cfg["renderer"]["scene_class_map"],mapping)

    def test_scene_depth_conversion_and_network_normalizer_are_distinct(self):
        raw={"id":"case","rgb":np.full((32,32,3),128,dtype=np.uint8),
             "depth":np.full((32,32),2.),"nir":None,"targets":[],
             "depth_unit_conversion":1000.,"depth_scale":1000.}
        cfg={"model":{"depth_scale":10000.}}
        prepared=_load_case(raw,cfg)
        self.assertEqual(prepared["depth_scale"],10000.)
        np.testing.assert_array_equal(prepared["depth"],np.full((32,32),2000.))
        again=_load_case(prepared,cfg)
        np.testing.assert_array_equal(again["depth"],prepared["depth"])
        np.testing.assert_array_equal(raw["depth"],np.full((32,32),2.))
        unknown=_load_case({**raw,"depth_unit_conversion":None},cfg)
        self.assertIsNone(unknown["depth"])
        self.assertEqual(unknown["depth_scale"],10000.)


@unittest.skipUnless(HAS_TORCH,"torch is optional outside training")
class RouterGradientTests(unittest.TestCase):
    def test_scene_conversion_uses_the_same_network_scale_as_dataset_inputs(self):
        from tefed.model import sample_tensor
        cfg={"model":{"depth_scale":10000.}}
        case={"id":"case","rgb":np.full((32,32,3),128,dtype=np.uint8),
              "depth":np.full((32,32),2.),"nir":None,"targets":[],"depth_unit_conversion":1000.}
        rendered=_load_case(case,cfg)
        dataset={**rendered,"depth":np.full((32,32),2000.),"depth_scale":10000.}
        rendered_tensor=sample_tensor(rendered)
        dataset_tensor=sample_tensor(dataset)
        import torch
        torch.testing.assert_close(rendered_tensor,dataset_tensor)
        self.assertAlmostEqual(float(rendered_tensor[0,3,0,0]),.2)

    def test_descriptor_only_bank_has_no_observation_gradient(self):
        d=source_descriptor();renderer=FakeRenderer();cfg=config_for(services(renderer))
        manifest=SimpleNamespace(calibration_ids=(),by_id={})
        result=update_router(FakeBank(),FakeRouter(),manifest,[],{d["source_state_key"]:STATE},
                             {"client1":STATE},calibration(),cfg,0,descriptors=[d])
        self.assertEqual(result["observation_probes"],0)
        self.assertEqual(result["descriptor_probes"],1)
        self.assertEqual(result["steps"],0)
        self.assertEqual(result["status"],"no_observation_gradient")

    def test_descriptor_probabilities_are_constants_with_complete_bank_scaling(self):
        import torch
        obs=torch.tensor([[1.,0,0,0,0,0]],requires_grad=True)
        desc=torch.tensor([[0.,1,0,0,0,0]],requires_grad=True)
        objective=router_objective(obs.softmax(-1),desc.softmax(-1),torch.tensor([2]),torch.tensor([1]),0.)
        objective["loss"].backward()
        grad=obs.grad.clone()
        self.assertIsNone(desc.grad)
        self.assertGreater(float(grad.abs().sum()),0.)
        obs2=obs.detach().clone().requires_grad_()
        objective2=router_objective(obs2.softmax(-1),torch.empty(0,6),torch.tensor([2]),torch.empty(0,dtype=torch.long),0.)
        objective2["loss"].backward()
        torch.testing.assert_close(grad,obs2.grad/2)

    def test_update_router_only_changes_router_and_uses_compatible_lowest_loss(self):
        import torch
        from torch import nn
        class Bank(nn.Module):
            def __init__(self):
                super().__init__();self.weight=nn.Parameter(torch.ones(1))
        class Router(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=nn.Linear(17,6);self.register_buffer("prototypes",torch.tensor(PROTOTYPES,dtype=torch.float32))
            def forward(self,x):
                return self.encoder(x).softmax(-1)
        bank=Bank();router=Router()
        record=SimpleNamespace(id="v",client_id="client1",split="validation")
        manifest=SimpleNamespace(calibration_ids=("v",),by_id={"v":record})
        renderer=FakeRenderer();svc=services(renderer)
        # Use diagnostics for this fake bank.
        def assess(*args,**kwargs):
            p=prediction();return {"candidates":[{**p,"id":"anchor","route":[0,1,2],"weights":[.5,.3,.2],"routing_vector":[.1]*17}],"vectors":[[.1]*17]}
        svc["assess_frame"]=assess
        svc["load_sample"]=lambda record,size: {"id":"v","rgb":np.full((64,64,3),128,dtype=np.uint8),"targets":[{"box":[.2,.2,.6,.6],"class_id":0}]}
        svc["native_losses"]=lambda *args: {0:4.,1:3.,2:2.,3:1.,4:5.,5:6.}
        cfg=config_for(svc);cfg["router"]={"steps":2,"learning_rate":.01}
        initial_router=deepcopy(router.state_dict());initial_bank=bank.weight.detach().clone()
        result=update_router(bank,router,manifest,[],{}, {"client1":STATE},calibration(),cfg,1)
        self.assertEqual(result["status"],"updated")
        self.assertEqual(result["steps"],2)
        self.assertEqual(result["observation_probes"],3)
        self.assertTrue(all(p["target"]==3 for p in result["observation_targets"]))
        self.assertTrue(any(not torch.equal(v,router.state_dict()[k]) for k,v in initial_router.items()))
        torch.testing.assert_close(bank.weight,initial_bank)
        self.assertIsNone(bank.weight.grad)


if __name__=="__main__":
    unittest.main()
