"""Hand-calculated decision and boundary tests."""
from copy import deepcopy
from hashlib import sha256
import json
from math import log
import unittest

import numpy as np

from tefed.core import (
    aggregate_updates, allocate_rendering, associate_hypothesis, attribute_bins,
    candidate_reliability, case_compatibility, descriptor_fidelity,
    descriptor_projection, expert_agreement, fidelity_mask, fuse_candidate, iou,
    make_descriptor, modality_agreement, qualify_candidate, score_descriptors,
    select_reuse, select_route, threshold_for_state, validate_descriptor,
    validation_loss_score, variant_seed, verification_posterior, verify_case,
)
from tefed.state import (
    initialize_state, state_demand, state_specialist_probabilities,
    update_state, validate_state,
)


def detection(p=(.9,.1),box=(.1,.1,.3,.3),index=0):
    return {"box":list(box),"posterior":list(p),"index":index}


STATE={"q":.8,"p":.7,"h":[.2]*5}
PROTOTYPES={str(i+1):np.eye(5)[i].tolist() for i in range(5)}
PROTOTYPES["0"]=(np.ones(5)/np.sqrt(5)).tolist()


def descriptor(attrs=None,flags=None,source="image1",candidate_id="c1",round_id=0):
    candidate={"disposition":"correction","id":candidate_id,"class_id":0,
               "route":[0,1,2],"weights":[.5,.3,.2]}
    return make_descriptor(candidate,attrs or [.5]*8,[1]*8,flags or [1]*5,
                           "client1",source,round_id,STATE,"test-secret")


def model_state(backbone=0.,heads=0.):
    return {"backbone":{"w":np.array([backbone],dtype=float)},
            "heads":{str(i):{"w":np.array([heads],dtype=float)} for i in range(6)}}


class CoreDecisionTests(unittest.TestCase):
    def test_iou_and_stable_geometric_association(self):
        self.assertAlmostEqual(iou([0,0,.5,.5],[.25,0,.75,.5]),1/3)
        self.assertEqual(iou([0,0,0,0],[0,0,1,1]),0)
        a=detection((.8,.2),index=9)
        b=detection((.9,.1),index=3)
        c=detection((.9,.1),index=2)
        self.assertEqual(associate_hypothesis([a,b,c],a["box"])["index"],2)
        other_class=detection((.1,.9))
        self.assertIsNone(associate_hypothesis([other_class],a["box"],class_id=0))
        self.assertEqual(associate_hypothesis([other_class],a["box"])["class_id"],1)

    def test_general_always_active_top2_tie(self):
        selected=select_route([.1,.18,.18,.18,.18,.18])
        self.assertEqual(selected["route"],[0,1,2])
        self.assertAlmostEqual(sum(selected["weights"]),1)
        with self.assertRaises(ValueError):
            select_route([0,0,0,0,0,1])

    def test_original_mass_confidence_and_agreement(self):
        fused=fuse_candidate(detection(),{},[0,1,2],[.5,.3,.2])
        self.assertAlmostEqual(fused["posterior"][0],.9)
        self.assertAlmostEqual(fused["C"],.45)
        self.assertEqual(fused["AE"],0)
        outputs={"0":detection(),"1":detection(),"2":None}
        self.assertAlmostEqual(expert_agreement(outputs,[0,1,2],[.5,.3,.2]),.15/.31)
        fused=fuse_candidate(detection(),{"1":[detection((.7,.3))]},[0,1,2],[.5,.3,.2])
        self.assertAlmostEqual(fused["posterior"][0],.825)
        self.assertAlmostEqual(fused["C"],.66)

    def test_class_conflict_modality_retains_weight(self):
        modalities={"RGB":detection(),"D":detection((.1,.9)),"NIR":None}
        self.assertAlmostEqual(modality_agreement(detection(),modalities),.4/.75)
        self.assertEqual(modality_agreement(detection(),{"RGB":None}),0)

    def test_entropy_is_not_only_disagreement(self):
        cues={"C":1.,"AE":1.,"AM":1.,"G":1.,"S":1.}
        scored=candidate_reliability(cues,[1/3]*3)
        self.assertAlmostEqual(scored["D_E"],1.)
        self.assertAlmostEqual(scored["R"],.99)
        cues["AE"]=0.
        self.assertAlmostEqual(candidate_reliability(cues,[1/3]*3)["D_E"],2.)

    def test_explicit_cross_modal_ablation_omits_measurement(self):
        cues={"C":.8,"AE":.7,"G":.9,"S":.6}
        result=candidate_reliability(cues,[1/3]*3,{"omit_cross_modal":True})
        expected=(.22*.8+.24*.7+.21*.9+.24*.6-.08*(1-.7+1))/.91
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["R"],expected)
        self.assertFalse(candidate_reliability(cues,[1/3]*3)["valid"])

    def test_invalid_cue_never_fabricated(self):
        for missing in (None,float("nan"),float("inf"),-.01,1.01,True):
            candidate={"C":.8,"AE":.9,"AM":missing,"G":.8,"S":.8,
                       "route":[0,1,2],"weights":[.5,.3,.2]}
            result=qualify_candidate(candidate,STATE)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["R"])
            self.assertEqual(result["disposition"],"suppressed")

    def test_partition_threshold_first_and_eligibility(self):
        good={"C":1.,"AE":1.,"AM":1.,"G":1.,"S":1.,"route":[0,1,2],"weights":[1/3]*3}
        self.assertEqual(qualify_candidate(good,STATE)["disposition"],"qualified")
        correction={**good,"C":.5,"AE":.1,"AM":0.,"G":.5,"S":0.}
        result=qualify_candidate(correction,STATE)
        self.assertEqual(result["disposition"],"correction")
        self.assertEqual(qualify_candidate({**correction,"C":.4999},STATE)["disposition"],"suppressed")
        self.assertEqual(qualify_candidate({**correction,"G":.4999},STATE)["disposition"],"suppressed")

    def test_threshold_edges_and_legal_minimum(self):
        self.assertAlmostEqual(threshold_for_state({"q":1.,"p":1.,"h":[.2]*5}),.706)
        below=threshold_for_state({**STATE,"q":.548-1e-8})
        at=threshold_for_state({**STATE,"q":.548})
        self.assertAlmostEqual(below-at,.004)

    def test_descriptor_keys_masks_and_source_zero(self):
        d=descriptor()
        self.assertEqual(d["source_class"],0)
        self.assertEqual(d,descriptor())
        next_d=descriptor(round_id=1)
        self.assertEqual(d["source"],next_d["source"])
        self.assertNotEqual(d["source_state_key"],next_d["source_state_key"])
        self.assertEqual(attribute_bins([0]*8),[0]*8)
        self.assertEqual(attribute_bins([1]*8),[3]*8)
        mask=fidelity_mask([0]*8,[1]*8,[1,0,0,0,0])
        self.assertEqual(mask,[0,0,0,1,0,0,0,0])
        with self.assertRaises(ValueError):
            validate_descriptor(d,{})
        self.assertEqual(validate_descriptor(d,{d["source_state_key"]:STATE})["id"],d["id"])
        bad=deepcopy(d);bad["mask"]=[0]*8
        with self.assertRaises(ValueError):
            validate_descriptor(bad)

    def test_zero_mask_projection_relevance_and_lookup_requirement(self):
        d=descriptor(attrs=[0]*8,flags=[0]*5)
        self.assertEqual(descriptor_projection(d),[0.]*5)
        self.assertEqual(validation_loss_score(d,None),0.)
        scored=score_descriptors([d],["client1"],PROTOTYPES,None)[0]
        self.assertAlmostEqual(scored["priority_terms"]["relevance"],.5)
        with self.assertRaises(ValueError):
            score_descriptors([descriptor()],["client1"],PROTOTYPES,None)

    def test_recurrence_counts_clients_not_descriptors(self):
        d1=descriptor(candidate_id="a")
        d2=descriptor(candidate_id="b")
        d3=descriptor(candidate_id="c");d3["client_id"]="client2"
        lookup={"base":{family:[.1]*4 for family in ("illum","occ","ctx","mat","sens")}}
        result=score_descriptors([d1,d2,d3],["client1","client2","client3"],PROTOTYPES,lookup)
        for d in result:
            self.assertAlmostEqual(d["priority_terms"]["recurrence"],2/3)
        permuted=score_descriptors([d3,d2,d1],["client3","client2","client1"],PROTOTYPES,lookup)
        self.assertEqual({d["id"]:d["priority"] for d in result},{d["id"]:d["priority"] for d in permuted})

    def test_budget_hamilton_and_reproducible_seed(self):
        ds=[]
        for i in range(3):
            d=descriptor(candidate_id=str(i));d["priority"]=.85-i*.1;d["assignment_probabilities"]=[.2]*5;ds.append(d)
        result=allocate_rendering(ds,budget=9)
        self.assertEqual(result["used"],9)
        self.assertEqual([x["realized"] for x in result["audit"]],[7,2,0])
        self.assertEqual(sum(result["audit"][0]["counts"]),7)
        self.assertEqual(allocate_rendering(ds,0)["allocations"],[])
        self.assertEqual(allocate_rendering(ds,9),allocate_rendering(list(reversed(ds)),9))
        seed=variant_seed(1,0,"d",2,3)
        self.assertEqual(seed,int.from_bytes(sha256(b"1\x1f0\x1fd\x1f2\x1f3").digest()[:4],"big"))
        self.assertNotEqual(seed,variant_seed(1,1,"d",2,3))

    def test_coverage_before_renormalization(self):
        failed=verification_posterior({0:[.9,.1]},[0,1,2],[.5,.3,.2])
        self.assertFalse(failed["passed"])
        self.assertAlmostEqual(failed["coverage"],.5)
        self.assertIsNone(failed["posterior"])
        passed=verification_posterior({0:[.9,.1],1:[.7,.3]},[0,1,2],[.5,.3,.2])
        self.assertTrue(passed["passed"])
        self.assertAlmostEqual(passed["coverage"],.8)
        self.assertAlmostEqual(passed["posterior"][0],.825)

    def test_fidelity_one_active_measurement_and_missing_failure(self):
        attrs=[0]*8;attrs[3]=.5
        d=descriptor(attrs=attrs,flags=[0]*5)
        measured=[None]*8;measured[3]=.38-.22*.4
        result=descriptor_fidelity(measured,d)
        self.assertAlmostEqual(result["fidelity"],.6)
        measured[3]=None
        self.assertFalse(descriptor_fidelity(measured,d)["valid"])
        no_mask=descriptor(attrs=[0]*8,flags=[0]*5)
        self.assertFalse(descriptor_fidelity([None]*8,no_mask)["valid"])

    def test_verification_class_check_follows_geometry_association(self):
        d=descriptor()
        variant={"id":"v","target_box":[.1,.1,.3,.3],"attrs_measured":d["attrs"],"geometry":1.}
        outputs={str(i):detection() for i in (0,1,2)}
        self.assertTrue(verify_case(variant,d,outputs)["retained"])
        opposite={str(i):detection((.1,.9)) for i in (0,1,2)}
        result=verify_case(variant,d,opposite)
        self.assertAlmostEqual(result["coverage"],1.)
        self.assertEqual(result["reason"],"margin")
        self.assertLess(result["margin"],0)
        self.assertEqual(verify_case(variant,d,{"0":detection()})["reason"],"coverage")

    def test_next_round_and_current_state_selection(self):
        demand=state_demand(STATE)
        pool=[]
        for i in range(9):
            attrs=demand.copy();attrs[0]=.20+.04*i
            d=descriptor(attrs=attrs,source=f"im{i}",candidate_id=f"c{i}")
            d["priority"]=.7
            pool.append({"id":f"v{i}","round":0,"d":d,"fidelity":1.})
        self.assertEqual(select_reuse(pool,STATE,0)["selected"],[])
        chosen=select_reuse(pool,STATE,1)
        self.assertLessEqual(len(chosen["selected"]),6)
        counts={}
        for item in chosen["selected"]:
            p=item["d"]["pattern"];counts[p]=counts.get(p,0)+1
        self.assertTrue(all(n<=2 for n in counts.values()))
        self.assertTrue(any(a["reason"]=="pattern_diversity" for a in chosen["audit"]))
        self.assertEqual(chosen,select_reuse(list(reversed(pool)),STATE,1))
        zero=select_reuse(pool,STATE,1,{"reuse_cap":0})
        self.assertEqual(zero["selected"],[])
        case=pool[0];before=deepcopy(case)
        other={"q":.1,"p":.1,"h":[.8,.05,.05,.05,.05]}
        self.assertNotEqual(case_compatibility(case,STATE)["compatibility"],case_compatibility(case,other)["compatibility"])
        self.assertEqual(case,before)

    def test_state_updates_have_no_same_round_leak(self):
        stats={"q_rgb":.2,"valid_depth_fraction":.8,"q_nir":.1,"mean_am_pre":.7,
               "assessed_count":10,"qualified_count":4,"mean_r":.6,"mean_s":.5,"mean_ae":.7,
               "failure_counts":[1,2,0,0,0]}
        original=deepcopy(STATE)
        updated=update_state(STATE,stats)
        self.assertEqual(STATE,original)
        self.assertAlmostEqual(updated["q"],.8)
        raw=.25*.4+.35*.6+.2*.5+.2*.7
        self.assertAlmostEqual(updated["p"],.6*.7+.4*raw)
        self.assertEqual(updated["h"],[2/8,3/8,1/8,1/8,1/8])
        self.assertAlmostEqual(initialize_state(stats)["p"],raw)
        empty={**stats,"assessed_count":0,"qualified_count":0,"failure_counts":[0]*5}
        self.assertEqual(update_state(STATE,empty)["p"],STATE["p"])
        self.assertEqual(update_state(STATE,empty)["h"],[.2]*5)
        with self.assertRaises(ValueError):
            validate_state({"q":.5,"p":.5,"h":[.1]*5})
        probabilities=state_specialist_probabilities(STATE)
        self.assertEqual(set(probabilities),set("12345"))
        self.assertAlmostEqual(sum(probabilities.values()),1.)

    def test_aggregation_bar_r_and_backbone_once(self):
        base=model_state()
        delta=model_state(backbone=10.,heads=2.)
        upload={"client_id":"a","delta":delta,"bar_R":[1.]*6,"gain":[1.]*6}
        result=aggregate_updates(base,[upload],{"a":STATE},PROTOTYPES)
        self.assertAlmostEqual(float(result["state"]["backbone"]["w"][0]),2.,places=6)
        self.assertAlmostEqual(float(result["state"]["heads"]["1"]["w"][0]),.4,places=6)
        correction_only={**upload,"bar_R":[0.]*6}
        zero=aggregate_updates(base,[correction_only],{"a":STATE},PROTOTYPES)
        self.assertEqual(zero["state"]["backbone"]["w"].tolist(),[0.])
        self.assertTrue(all(x==0 for x in zero["masses"][0]))
        inactive=model_state(backbone=10.,heads=0.)
        zero_head=aggregate_updates(base,[{**upload,"delta":inactive}],{"a":STATE},PROTOTYPES)
        self.assertEqual(zero_head["backbone_masses"],[0.])

    def test_directional_conflict_uses_previous_completed_update(self):
        base=model_state(backbone=1.,heads=1.)
        previous=model_state()
        upload={"client_id":"a","delta":model_state(backbone=-1.,heads=-1.),"bar_R":[1.]*6,"gain":[1.]*6}
        result=aggregate_updates(base,[upload],{"a":STATE},PROTOTYPES,previous_state=previous)
        for details in result["details"][0]["experts"]:
            self.assertAlmostEqual(details["conflict"],1.)
        for mass in result["masses"][0]:
            self.assertAlmostEqual(mass,0.)
        first=aggregate_updates(base,[upload],{"a":STATE},PROTOTYPES)
        self.assertTrue(all(d["conflict"]==0 for d in first["details"][0]["experts"]))

    def test_json_serializable_decisions(self):
        d=descriptor()
        d["priority"]=.5;d["assignment_probabilities"]=[.2]*5
        output=allocate_rendering([d],budget=2)
        json.dumps(output,allow_nan=False)
        result=verify_case({"target_box":[.1,.1,.3,.3],"geometry":1.,"attrs_measured":[.5]*8},d,{str(i):detection() for i in (0,1,2)})
        json.dumps(result,allow_nan=False)


if __name__=="__main__":
    unittest.main()
