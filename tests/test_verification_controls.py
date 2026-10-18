"""Frozen-route comparisons keep variants and predictions fixed."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analyze_route_controls import main
from tefed.verification_controls import compare_frozen_verification
from tests.test_core import STATE,descriptor,detection


class FrozenVerificationTests(unittest.TestCase):
    def inputs(self):
        d=descriptor()
        prediction=detection()
        frozen={"descriptor":d,"target_box":prediction["box"],"target_class":0,
            "attrs_measured":[.5]*8,"geometry":1.,
            "expert_outputs":{str(m):[deepcopy(prediction)] for m in range(6)},
            "current_reference":{"route":[0,3,4],"weights":[.4,.35,.25]},
            "cyclic_reference":{"route":[0,4,5],"weights":[.6,.25,.15],
                "profile":"candidate-route-tuple-shift11-v1","shift":11,
                "population_size":12,"recipient_position":0,"donor_position":11,
                "ordering_fields":["manifest_image_id","candidate_id"],"ordering_sha256":"a"*64,
                "run_seed":8,"round_index":0,"recipient_descriptor_id":d["id"],
                "recipient_candidate_id":"c1","donor_client_id":"client1","donor_local_identity":"donor-source",
                "donor_candidate_id":"c12","donor_source_observation_key":"donor-source-key"}}
        return [{"case_id":"case-a","entered_verification":True,"frozen_verification_inputs":frozen}],{d["source_state_key"]:STATE}

    def test_same_population_complete_tuples_and_micro_counts(self):
        rows,states=self.inputs();before=deepcopy(rows)
        result=compare_frozen_verification(rows,states,run_seed=8)
        self.assertTrue(result["complete"])
        for mode in ("stored","current","cyclic"):
            summary=result["summary"][mode]
            self.assertEqual(summary["rendered_count"],1)
            self.assertEqual(summary["retention_fraction"],1.)
            self.assertEqual(summary["retained_rta"],1.)
        reference=result["events"]["cyclic"][0]["reference"]
        self.assertEqual(reference["weights"],[.6,.25,.15])
        measured=result["events"]["cyclic"][0]["verification"]["fidelity_audit"]["measured_severities"]
        self.assertAlmostEqual(measured[1],.2)
        self.assertEqual(measured[0],.5)
        self.assertEqual(rows,before)

    def test_current_missing_anchor_is_recorded_as_failure(self):
        rows,states=self.inputs()
        rows[0]["frozen_verification_inputs"]["current_reference"]=None
        result=compare_frozen_verification(rows,states,run_seed=8)
        self.assertEqual(result["summary"]["current"]["retained_count"],0)
        self.assertEqual(result["summary"]["current"]["retention_fraction"],0.)
        self.assertIsNone(result["summary"]["current"]["retained_rta"])

    def test_missing_measurement_cannot_be_reported_as_complete(self):
        rows,states=self.inputs()
        rows[0]["frozen_verification_inputs"]["geometry"]=None
        with self.assertRaisesRegex(ValueError,"missing measurements"):
            compare_frozen_verification(rows,states,run_seed=8)
        result=compare_frozen_verification(rows,states,run_seed=8,allow_incomplete=True)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["summary"]["stored"]["retention_fraction"])

    def test_duplicate_case_missing_inputs_and_wrong_shift_rejected(self):
        rows,states=self.inputs()
        with self.assertRaisesRegex(ValueError,"unique"):
            compare_frozen_verification(rows*2,states,run_seed=8)
        with self.assertRaisesRegex(ValueError,"frozen verification inputs"):
            compare_frozen_verification([{"case_id":"a","entered_verification":True}],states,run_seed=8)
        rows[0]["frozen_verification_inputs"]["cyclic_reference"]["shift"]=1
        with self.assertRaisesRegex(ValueError,"shift protocol"):
            compare_frozen_verification(rows,states,run_seed=8)

    def test_renderer_target_geometry_must_match_compressed_reference(self):
        rows,states=self.inputs()
        rows[0]["frozen_verification_inputs"]["renderer_target"]={"box":[.2,.2,.5,.5],"class_id":0}
        with self.assertRaisesRegex(ValueError,"identical verification"):
            compare_frozen_verification(rows,states,run_seed=8)

    def test_cli_binds_completed_rounds_and_refuses_existing_output(self):
        rows,states=self.inputs()
        with TemporaryDirectory() as folder:
            root=Path(folder)
            files={"configuration.json":{"experiment":{"rounds":1,"seed":8}},"source_states.json":states,
                   "environment.json":{"seed":8},
                   "status.json":{"status":"completed","completed_rounds":1},
                   "verification_round_000.json":rows}
            for name,data in files.items():
                (root/name).write_text(json.dumps(data),encoding="utf-8")
            args=["--run",str(root),"--out",str(root/"controls.json")]
            self.assertEqual(main(args),0)
            result=json.loads((root/"controls.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["inputs"]),5)
            self.assertTrue(all(len(row["sha256"])==64 for row in result["inputs"]))
            with self.assertRaises(FileExistsError):
                main(args)
            (root/"status.json").write_text('{"status":"running","completed_rounds":1}')
            with self.assertRaisesRegex(ValueError,"completed run"):
                main(["--run",str(root),"--out",str(root/"other.json")])

    def test_cyclic_reference_binds_descriptor_round_seed_and_ordering(self):
        for key,value in (("recipient_descriptor_id","wrong"),("round_index",99),("run_seed",999),
                          ("ordering_fields",["wrong"]),("ordering_sha256","Z"*64)):
            with self.subTest(key=key):
                rows,states=self.inputs()
                rows[0]["frozen_verification_inputs"]["cyclic_reference"][key]=value
                with self.assertRaises(ValueError):
                    compare_frozen_verification(rows,states,run_seed=8)


if __name__=="__main__":
    unittest.main()
