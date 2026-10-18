"""Fidelity provenance and early-gate diagnostics."""
from copy import deepcopy
import json
from tempfile import TemporaryDirectory
import unittest

from tefed import core
from tefed.server import construct_cases
from tests.test_core import descriptor, detection
from tests.test_server import (FakeBank, FakeRenderer, FakeRouter, STATE,
    calibration, config_for, services, source_descriptor)


class FidelityAuditTests(unittest.TestCase):
    def test_fixed_mask_weighted_score_and_intervals_are_recoverable(self):
        d=descriptor()
        measured=[.1,.2,.3,.4,.5,.6,.7,.8]
        result=core.descriptor_fidelity(measured,d)
        audit=result["audit"]
        expected=[]
        for x,interval,tolerance in zip(measured,audit["requested_intervals"],audit["normalizers"]):
            expected.append(max(0.,1-max(interval[0]-x,0.,x-interval[1])/tolerance))
        mass=sum(w*m for w,m in zip(core.FIDELITY_WEIGHTS,d["mask"]))
        score=sum(w*f*m for w,f,m in zip(core.FIDELITY_WEIGHTS,expected,d["mask"]))/mass
        self.assertAlmostEqual(result["fidelity"],score)
        self.assertEqual(audit["active_weight_mass"],mass)
        self.assertEqual(audit["active_mask"],d["mask"])
        self.assertEqual(audit["attribute_names"],list(core.ATTRIBUTES))
        self.assertEqual(audit["attribute_fidelities"],expected)
        self.assertTrue(audit["evaluated"])
        self.assertEqual(audit["score"],result["fidelity"])

    def test_active_zero_is_measured_and_inactive_nonfinite_is_not_activated(self):
        d=source_descriptor()
        measured=[float("nan")]*8
        measured[3]=0.
        result=core.descriptor_fidelity(measured,d)
        self.assertTrue(result["valid"])
        self.assertEqual(result["fidelity"],1.)
        self.assertEqual(result["audit"]["measurement_status"][3],"measured")
        self.assertEqual(result["audit"]["measurement_status"][0],"inactive")
        self.assertIsNone(result["audit"]["measured_severities"][0])
        json.dumps(result,allow_nan=False)

    def test_missing_active_measurement_is_undefined_without_fabricated_score(self):
        d=source_descriptor()
        result=core.descriptor_fidelity([None]*8,d)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["fidelity"])
        self.assertEqual(result["audit"]["first_failed_gate"],"fidelity")
        self.assertEqual(result["audit"]["measurement_status"][3],"undefined")
        self.assertEqual(result["audit"]["active_weight_mass"],.12)

    def test_coverage_failure_does_not_claim_fidelity_was_evaluated(self):
        d=descriptor()
        variant={"id":"variant","target_box":[.1,.1,.3,.3],
                 "attrs_measured":[.5]*8,"geometry":1.}
        result=core.verify_case(variant,d,{"0":[detection()]})
        self.assertEqual(result["first_failed_gate"],"coverage")
        self.assertFalse(result["fidelity_audit"]["evaluated"])
        self.assertIsNone(result["fidelity_audit"]["score"])

    def test_final_gate_reason_is_attached_to_fidelity_audit(self):
        d=descriptor()
        variant={"id":"variant","target_box":[.1,.1,.3,.3],
                 "attrs_measured":[.5]*8,"geometry":1.}
        outputs={str(m):[detection((.1,.9))] for m in d["route"]}
        result=core.verify_case(variant,d,outputs)
        self.assertEqual(result["first_failed_gate"],"margin")
        self.assertTrue(result["fidelity_audit"]["evaluated"])
        self.assertEqual(result["fidelity_audit"]["first_failed_gate"],"margin")

    def test_server_early_failure_keeps_source_variant_and_fixed_mask(self):
        d=source_descriptor()
        before=deepcopy(d)
        renderer=FakeRenderer()
        service=services(renderer)
        service["predict_outputs"]=lambda *args:{"outputs":{"0":[detection(box=(.2,.2,.6,.6))]},"proposal_count":1}
        with TemporaryDirectory() as folder:
            result=construct_cases([d],FakeBank(),FakeRouter(),{d["source_state_key"]:STATE},
                ["client1"],calibration(),config_for(service),3,8,folder)
        entry=next(x for x in result["audit"] if x.get("entered_verification"))
        self.assertEqual(entry["first_failed_gate"],"coverage")
        self.assertEqual(entry["case_id"],entry["rendered_variant_id"])
        self.assertEqual(entry["creation_round"],3)
        self.assertEqual(entry["source_observation_key"],d["source"])
        self.assertEqual(entry["descriptor_id"],d["id"])
        self.assertEqual(entry["source_route"],d["route"])
        self.assertEqual(entry["fidelity_audit"]["active_mask"],d["mask"])
        self.assertEqual(entry["fidelity_audit"]["measured_severities"],[None]*8)
        self.assertFalse(entry["fidelity_audit"]["evaluated"])
        self.assertEqual(d,before)


if __name__=="__main__":
    unittest.main()
