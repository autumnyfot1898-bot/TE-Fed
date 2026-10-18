"""Cue extraction from image arrays and predictions."""
from copy import deepcopy
import json
from math import log
import unittest

import numpy as np

from tefed.cues import extract_cues, quality_stats, routing_vector
from tefed.core import make_descriptor
from tefed.state import update_state


def prediction(box=(.2,.2,.6,.6),posterior=(.9,.1),index=0):
    return {"box":list(box),"posterior":list(posterior),"index":index}


def sample(modal=True):
    result={"rgb":np.full((128,128,3),128,dtype=np.uint8),"proposal_count":40,
            "enabled_modalities":["RGB","D","NIR"] if modal else ["RGB"]}
    if modal:
        result.update(depth=np.ones((128,128)),nir=np.full((128,128),128,dtype=np.uint8))
    return result


def calibration(height=.4):
    return {"luminance_median":128/255,"depth_validation_median":1.,
            "class_priors":{"0":{"a_c":log(height),"b_c":-1.,"r_c_med":1.},
                            "1":{"a_c":log(height),"b_c":-1.,"r_c_med":1.}}}


STATE={"q":.8,"p":.7,"h":[.2]*5}


class CueMeasurementTests(unittest.TestCase):
    def test_calibrated_depth_geometry_and_attributes(self):
        s=sample();p=prediction()
        cues=extract_cues(s,p,{"RGB":[p],"D":[p],"NIR":[p]},[[p]],calibration())
        self.assertAlmostEqual(cues["G"],1.,places=5)
        self.assertAlmostEqual(cues["Gamma_pre"],1.,places=5)
        self.assertEqual(cues["AM"],1.)
        self.assertEqual(cues["S"],1.)
        self.assertEqual(cues["sigma_b"],0.)
        self.assertAlmostEqual(cues["attrs"][1],.2)
        self.assertAlmostEqual(cues["attrs"][3],0.)
        self.assertAlmostEqual(cues["attrs"][4],0.,places=5)
        self.assertAlmostEqual(cues["attrs"][5],0.,places=5)
        self.assertEqual(cues["attrs"][6],0.)
        self.assertEqual(cues["availability"],[1]*8)
        self.assertEqual(len(routing_vector(cues,STATE)),17)
        json.dumps(cues,allow_nan=False)

    def test_missing_calibration_remains_missing_and_no_fake_depth_prior(self):
        p=prediction();s=sample()
        cal={"task":"another_task","class_priors":{"0":{"a_c":None,"b_c":None,"r_c_med":None}}}
        cues=extract_cues(s,p,{"RGB":[p]},None,cal)
        self.assertIsNone(cues["G"])
        self.assertIsNone(cues["attrs"][3])
        self.assertIsNone(cues["attrs"][4])
        self.assertIsNone(cues["attrs"][5])
        self.assertEqual(cues["availability"][3:6],[0,0,0])
        with self.assertRaises(ValueError):
            routing_vector(cues,STATE)

    def test_unreliable_depth_uses_area_aspect_fallback(self):
        p=prediction();s=sample();s["depth"][:]=np.nan
        cues=extract_cues(s,p,{"RGB":[p]},[[p]],calibration())
        self.assertEqual(cues["measurements"]["depth_valid_pixels"],0)
        self.assertFalse(cues["measurements"]["depth_reliable"])
        self.assertAlmostEqual(cues["G"],1.)
        self.assertAlmostEqual(cues["b_view"],.3)
        self.assertAlmostEqual(cues["attrs"][5],.5)
        self.assertEqual(cues["attrs"][7],1.)
        self.assertEqual(cues["qstats"]["valid_depth_fraction"],0.)

    def test_missing_depth_calibration_does_not_pretend_measured_residual(self):
        p=prediction();s=sample();cal=calibration();del cal["depth_validation_median"]
        cues=extract_cues(s,p,{"RGB":[p]},[[p]],cal)
        self.assertIsNone(cues["measurements"]["depth_residual"])
        self.assertFalse(cues["measurements"]["depth_reliable"])
        self.assertAlmostEqual(cues["G"],1.)
        self.assertIn("depth_calibration_missing",cues["cue_audit"])
        self.assertIsNone(cues["attrs"][5])

    def test_class_inconsistent_modality_weight_is_not_reassigned(self):
        p=prediction();opposite=prediction(posterior=(.1,.9))
        cues=extract_cues(sample(),p,{"RGB":[p],"D":[opposite],"NIR":None},None,calibration())
        self.assertAlmostEqual(cues["AM_pre"],.4/.75)
        self.assertEqual(cues["S"],0.)
        self.assertIsNone(cues["attrs"][5])
        # Unavailable hypotheses carry no weight.
        unavailable=extract_cues(sample(),p,{"RGB":[p],"D":[],"NIR":None},None,calibration())
        self.assertEqual(unavailable["AM_pre"],1.)

    def test_temporal_match_threshold_and_augmentation_fallback(self):
        p=prediction(box=(0.,0.,.5,.5))
        half=prediction(box=(1/6,0.,2/3,.5))
        cues=extract_cues(sample(False),p,{"RGB":[p]},[[half]],calibration(height=.5))
        self.assertAlmostEqual(cues["S"],.5)
        self.assertAlmostEqual(cues["attrs"][5],.5)
        nonmatching=prediction(box=(.6,.6,.9,.9))
        previous=extract_cues(sample(False),p,{"RGB":[p]},
            {"previous":[nonmatching],"weak":[p],"strong":[p]},calibration(height=.5))
        self.assertEqual(previous["S"],0.)
        fallback=extract_cues(sample(False),p,{"RGB":[p]},
            {"previous":None,"weak":[p],"strong":[p]},calibration(height=.5))
        self.assertEqual(fallback["S"],1.)

    def test_context_real_measurement_and_unavailable_proposal_channel(self):
        p=prediction();s=sample(False)
        s["proposal_count"]=260
        cues=extract_cues(s,p,{"RGB":[p]},None,calibration())
        self.assertAlmostEqual(cues["attrs"][6],.55)
        self.assertEqual(cues["availability"][6],1)
        del s["proposal_count"]
        cues=extract_cues(s,p,{"RGB":[p]},None,calibration())
        self.assertIsNone(cues["attrs"][6])
        self.assertEqual(cues["availability"][6],0)
        tiny=prediction(box=(.4,.4,.41,.41))
        s["proposal_count"]=100
        cues=extract_cues(s,tiny,{"RGB":[tiny]},None,calibration())
        self.assertIsNone(cues["attrs"][6])

    def test_occlusion_excludes_self_and_lower_confidence_comparators(self):
        p=prediction();lower=prediction(posterior=(.8,.2),index=1)
        s=sample(False);s["foreground_detections"]=[p,lower]
        cues=extract_cues(s,p,{"RGB":[p]},None,calibration())
        self.assertAlmostEqual(cues["attrs"][2],0.)
        stronger=prediction(posterior=(.95,.05),index=2)
        s["foreground_detections"].append(stronger)
        cues=extract_cues(s,p,{"RGB":[p]},None,calibration())
        self.assertAlmostEqual(cues["attrs"][2],.85)

    def test_valid_zero_luminance_and_missing_crop_calibration(self):
        s=sample(False);s["rgb"][:]=0;p=prediction()
        cues=extract_cues(s,p,{"RGB":[p]},None,calibration())
        self.assertEqual(cues["attrs"][3],1.)
        self.assertEqual(cues["availability"][3],1)
        cal=calibration();cal["luminance_median"]=0.
        cues=extract_cues(s,p,{"RGB":[p]},None,cal)
        self.assertIsNone(cues["attrs"][3])
        self.assertEqual(cues["rule_flags"][0],0)

    def test_disabled_sensing_none_and_full_frame_no_candidate_statistics(self):
        s=sample(False)
        measured=quality_stats(s)
        self.assertAlmostEqual(measured["q_rgb"],.7)
        self.assertIsNone(measured["valid_depth_fraction"])
        self.assertIsNone(measured["q_nir"])
        self.assertIsNone(measured["mean_am_pre"])
        stats={**measured,"assessed_count":0,"qualified_count":0,"failure_counts":[0]*5}
        next_state=update_state(STATE,stats)
        self.assertAlmostEqual(next_state["q"],.3)
        self.assertEqual(next_state["p"],STATE["p"])
        cues=extract_cues(s,prediction(),{"RGB":[prediction()]},None,calibration())
        self.assertIsNone(cues["attrs"][7])
        self.assertEqual(cues["availability"][7],0)
        s["enabled_modalities"]=["RGB","D","NIR"]
        failed=quality_stats(s)
        self.assertEqual(failed["valid_depth_fraction"],0.)
        self.assertEqual(failed["q_nir"],1.)

    def test_no_ground_truth_access_and_none_descriptor_mask(self):
        class Poison:
            def __iter__(self):
                raise AssertionError("ground truth accessed")
        s=sample(False);s["targets"]=Poison();p=prediction()
        cues=extract_cues(s,p,{"RGB":[p]},None,calibration())
        candidate={"id":"x","class_id":0,"disposition":"correction","route":[0,1,2],"weights":[.5,.3,.2]}
        d=make_descriptor(candidate,cues["attrs"],cues["availability"],cues["rule_flags"],"c","image",0,STATE,"key")
        self.assertEqual(d["attrs"][5],0.)
        self.assertEqual(d["mask"][5],0)
        self.assertEqual(d["availability"][5],0)

    def test_registered_ir_manifest_alias_keeps_auxiliary_evidence(self):
        s=sample(False);s["nir"]=np.full((128,128),128,dtype=np.uint8)
        s["enabled_modalities"]=["RGB","IR"]
        measured=quality_stats(s)
        self.assertEqual(measured["q_nir"],0.)
        self.assertIsNone(measured["valid_depth_fraction"])
        p=prediction();other=prediction(posterior=(.1,.9))
        cues=extract_cues(s,p,{"RGB":[p],"NIR":[other]},None,calibration())
        self.assertAlmostEqual(cues["AM_pre"],.4/(.4+.25))
        self.assertEqual(cues["availability"][7],1)

    def test_inputs_not_mutated_and_unknown_class_prior_rejected(self):
        s=sample();p=prediction();original=s["rgb"].copy();cal=calibration();before=deepcopy(cal)
        extract_cues(s,p,{"RGB":[p]},None,cal)
        np.testing.assert_array_equal(s["rgb"],original)
        self.assertEqual(cal,before)
        unknown={"box":p["box"],"posterior":[.1,.1,.8]}
        cues=extract_cues(s,unknown,{"RGB":[unknown]},None,cal)
        self.assertIsNone(cues["G"])


if __name__=="__main__":
    unittest.main()
