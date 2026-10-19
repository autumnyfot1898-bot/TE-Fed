"""Hand-calculated analysis fixtures."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tefed.analysis import (ENERGY_NODES,analyze_run_dirs,compare_exposure,
    factorial_contrasts,integrate_round_energy,latency_p95_worst_client,
    read_power_csv,summarize_seed_metric)
from tefed.analysis import case_event_proportions
from scripts.record_power import read_nvidia,read_sysfs


def ledger(a=1,b=3,round_id=0):
    return [{"round_index":round_id,"clients":{
        "client1":{"optimizer_steps":2,"presentations":{"correction":a},"optimizer_steps_semantics":"actual_applied_updates"},
        "client2":{"optimizer_steps":2,"presentations":{"correction":b},"optimizer_steps_semantics":"actual_applied_updates"}},
        "reuse_selected_ids":{"client1":["a"],"client2":["b"]},"reporting_weighted_mean":7.}]


def power_rows():
    return [{"node":node,"ts":float(t),"power_w":10.+2*t} for node in ENERGY_NODES for t in range(3)]


def write_run(path,seed):
    path.mkdir()
    environment={"seed":seed,"manifest_sha256":"a"*64,"calibration_sha256":"b"*64,
                 "initial_checkpoint_sha256":"c"*64,"versions":{"torch":"fixture"}}
    configuration={"task":"tomato","_config_path":f"/local/{seed}/config.toml",
        "experiment":{"method":"fixture_method","seed":seed,"seeds":[seed],"rounds":1,
                      "output":str(path),"manifest":f"/local/{seed}/manifest.json",
                      "calibration":f"/local/{seed}/calibration.json"},
        "model":{"classes":2,"initial_checkpoint":f"/local/{seed}/initial.pt"},
        "decision":{"render_budget":1152},"loss":{"beta":.5},
        "optimizer":{"learning_rate":.01},"router":{"steps":20},
        "renderer":{"enabled":True,"scene":"/fixed/scene.blend"},
        "evaluation":{"views":["hard"]}}
    (path/"environment.json").write_text(json.dumps(environment),encoding="utf-8")
    (path/"configuration.json").write_text(json.dumps(configuration),encoding="utf-8")
    (path/"calibration.json").write_text(json.dumps({"data_origin":"software_fixture","T_C":1.}),encoding="utf-8")
    (path/"metrics.json").write_text(json.dumps({"hard":{"mAP50":.2+.1*seed,"AP75":.1+.1*seed,"mAP50_95":.15+.1*seed}}),encoding="utf-8")
    (path/"rounds.jsonl").write_text(json.dumps(ledger()[0])+"\n",encoding="utf-8")
    return path


class AnalysisTests(unittest.TestCase):
    def test_ten_seed_mean_sample_sd_and_strict_seed_set(self):
        rows=[{"seed":i,"value":float(i+1)} for i in range(10)]
        result=summarize_seed_metric(rows)
        self.assertEqual(result["mean"],5.5)
        self.assertAlmostEqual(result["sample_sd"],np.sqrt(82.5/9))
        for bad in (rows[:-1],rows+[rows[0]],rows+[{"seed":10,"value":3.}],rows[:-1]+[{"seed":9,"value":None}]):
            with self.assertRaises(ValueError):
                summarize_seed_metric(bad)

    def test_client_round_not_weighted_or_global_mean_matching(self):
        left,right=ledger(1,3),ledger(3,1)
        self.assertEqual(sum(r["presentations"]["correction"] for r in left[0]["clients"].values()),sum(r["presentations"]["correction"] for r in right[0]["clients"].values()))
        result=compare_exposure(left,right)
        self.assertFalse(result["matched"])
        self.assertFalse(result["reporting_weights_used"])
        self.assertEqual(len(result["comparisons"]),2)
        self.assertTrue(compare_exposure(left,deepcopy(left))["matched"])
        changed=deepcopy(left);changed[0]["clients"]["client1"]["optimizer_steps"]+=1
        self.assertFalse(compare_exposure(left,changed)["matched"])
        changed=deepcopy(left);changed[0]["reuse_selected_ids"]["client1"].append("c")
        self.assertFalse(compare_exposure(left,changed)["matched"])

    def test_incomplete_or_duplicate_execution_ledgers_fail(self):
        left=ledger();right=ledger()
        del right[0]["clients"]["client1"]["presentations"]["correction"]
        with self.assertRaises(ValueError):
            compare_exposure(left,right)
        with self.assertRaises(ValueError):
            compare_exposure(left,left+left)
        missing=deepcopy(left);del missing[0]["clients"]["client2"]
        result=compare_exposure(left,missing)
        self.assertFalse(result["matched"])
        self.assertEqual(result["missing_right"],[["client2",0]])

    def test_factorial_within_seed_polynomial_oracle(self):
        rows=[]
        for seed in range(10):
            for a in (0,1):
                for b in (0,1):
                    for c in (0,1):
                        value=10+seed+2*a+3*b+4*c+5*a*b+6*a*c+7*b*c+8*a*b*c
                        rows.append({"seed":seed,"factors":{"A":a,"B":b,"C":c},"value":value})
        result=factorial_contrasts(rows)
        expected={"A":9.5,"B":11.,"C":12.5,"AB":9.,"AC":10.,"BC":11.,"ABC":8.}
        for name,value in expected.items():
            self.assertAlmostEqual(result["summary"][name]["mean"],value)
            self.assertAlmostEqual(result["summary"][name]["sample_sd"],0.)
        with self.assertRaises(ValueError):
            factorial_contrasts(rows[:-1])
        with self.assertRaises(ValueError):
            factorial_contrasts(rows+[rows[0]])
        self.assertIn("none",result["execution_inference"])

    def test_trapezoids_interpolate_boundaries_and_require_all_five_devices(self):
        result=integrate_round_energy(power_rows(),[{"round_index":0,"start_ts":0.,"end_ts":2.}],expected_interval_s=1.)
        self.assertAlmostEqual(result["total_energy_j"],120.)
        self.assertTrue(all(v==24. for v in result["rounds"][0]["node_energy_j"].values()))
        inner=integrate_round_energy(power_rows(),[{"round_index":0,"start_ts":.5,"end_ts":1.5}],expected_interval_s=1.)
        self.assertAlmostEqual(inner["total_energy_j"],60.)
        missing=[r for r in power_rows() if r["node"]!="J4"]
        with self.assertRaises(ValueError):
            integrate_round_energy(missing,[{"round_index":0,"start_ts":0.,"end_ts":2.}],expected_interval_s=1.)

    def test_power_missing_point_clock_reversal_boundary_and_unit_failures(self):
        rounds=[{"round_index":0,"start_ts":0.,"end_ts":2.}]
        for rows in ([r for r in power_rows() if not(r["node"]=="J1" and r["ts"]==1)],
                     [{**r,"ts":-.1 if r["node"]=="J1" and r["ts"]==1 else r["ts"]} for r in power_rows()],
                     [r for r in power_rows() if not(r["node"]=="J2" and r["ts"]==0)]):
            with self.assertRaises(ValueError):
                integrate_round_energy(rows,rounds,expected_interval_s=1.)
        with self.assertRaises(ValueError):
            integrate_round_energy(power_rows(),rounds)
        with self.assertRaises(ValueError):
            integrate_round_energy(power_rows(),[{"round_index":0,"seconds":2.}],expected_interval_s=1.)
        with self.assertRaises(ValueError):
            integrate_round_energy(power_rows(),rounds+[{"round_index":1,"start_ts":1.,"end_ts":2.}],expected_interval_s=1.)

    def test_worst_node_p95_is_not_pooled_p95(self):
        rows=[{"node":f"J{node+1}","latency_ms":float(i+100*node)} for node in range(4) for i in range(200)]
        result=latency_p95_worst_client(rows)
        self.assertEqual(result["worst_node"],"J4")
        self.assertAlmostEqual(result["worst_node_p95_ms"],489.05)
        self.assertNotAlmostEqual(result["worst_node_p95_ms"],float(np.percentile([r["latency_ms"] for r in rows],95)))
        with self.assertRaises(ValueError):
            latency_p95_worst_client(rows[:-1])

    def test_raw_power_csv_requires_explicit_scope_and_units(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/"gpu.csv"
            p.write_text("ts,power_w,memory_mib,node\n0,10,32,serverGPU\n1,20,34,serverGPU\n",encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                read_power_csv([p])
            manifest={"node":"serverGPU","power_unit":"W","time_unit":"unix_utc_seconds","scope":"server_gpu","status":"completed"}
            side=p.with_suffix(".csv.manifest.json")
            side.write_text(json.dumps(manifest),encoding="utf-8")
            self.assertEqual(len(read_power_csv([p])["rows"]),2)
            manifest["power_unit"]="mW";side.write_text(json.dumps(manifest),encoding="utf-8")
            with self.assertRaises(ValueError):
                read_power_csv([p])

    def test_read_only_hardware_readers_do_not_replace_unavailable_power(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/"power_input";p.write_text("12500",encoding="ascii")
            self.assertAlmostEqual(read_sysfs(p,"mW")["power_w"],12.5)
            self.assertAlmostEqual(read_sysfs(p,"uW")["power_w"],.0125)
            with self.assertRaises(FileNotFoundError):
                read_sysfs(p.with_name("missing"),"mW")
        with patch("scripts.record_power.subprocess.run",return_value=subprocess.CompletedProcess([],0,"12.5, 128\n","")):
            self.assertEqual(read_nvidia(),{"power_w":12.5,"memory_mib":128.})
        with patch("scripts.record_power.subprocess.run",return_value=subprocess.CompletedProcess([],0,"[N/A], 128\n","")):
            with self.assertRaises(ValueError):
                read_nvidia()

    def test_run_analysis_allows_seed_output_and_bound_paths_but_requires_seed_set(self):
        with TemporaryDirectory() as folder:
            paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
            result=analyze_run_dirs(paths,expected_seeds=(0,1))
            hard=next(r for r in result["summary"] if r["metric"]=="mAP50")
            self.assertAlmostEqual(hard["mean"],.25)
            self.assertEqual(result["runs"][0]["protocol_sha256"],result["runs"][1]["protocol_sha256"])
            self.assertFalse(result["historical_results_embedded"])
            with self.assertRaises(ValueError):
                analyze_run_dirs(paths)

    def test_seed_summary_rejects_mixed_protocol_settings(self):
        changes=[("decision","render_budget",576),("loss","beta",0.),
                 ("model","classes",5),("optimizer","learning_rate",.02),
                 ("router","steps",10),("renderer","enabled",False),
                 ("evaluation","views",["participating"]),("renderer","scene","/other/scene.blend")]
        for section,key,value in changes:
            with self.subTest(section=section,key=key), TemporaryDirectory() as folder:
                paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
                file=paths[1]/"configuration.json"
                config=json.loads(file.read_text(encoding="utf-8"));config[section][key]=value
                file.write_text(json.dumps(config),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"fixed protocol"):
                    analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_seed_summary_retains_task_and_unknown_configuration_options(self):
        for patch in ({"task":"dronevehicle"},{"custom_control":{"enabled":True}}):
            with self.subTest(patch=patch), TemporaryDirectory() as folder:
                paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
                file=paths[1]/"configuration.json"
                config=json.loads(file.read_text(encoding="utf-8"));config.update(patch)
                file.write_text(json.dumps(config),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"fixed protocol"):
                    analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_seed_summary_rejects_different_source_artifact_identities(self):
        for key in ("manifest_sha256","calibration_sha256","initial_checkpoint_sha256"):
            with self.subTest(key=key), TemporaryDirectory() as folder:
                paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
                file=paths[1]/"environment.json"
                environment=json.loads(file.read_text(encoding="utf-8"));environment[key]="d"*64
                file.write_text(json.dumps(environment),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"fixed protocol"):
                    analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_seed_summary_requires_hashes_for_bound_artifact_paths(self):
        for key in ("manifest_sha256","calibration_sha256","initial_checkpoint_sha256"):
            for value in (None,"not-a-digest"):
                with self.subTest(key=key,value=value), TemporaryDirectory() as folder:
                    paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
                    file=paths[1]/"environment.json"
                    environment=json.loads(file.read_text(encoding="utf-8"));environment[key]=value
                    file.write_text(json.dumps(environment),encoding="utf-8")
                    with self.assertRaisesRegex(ValueError,key):
                        analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_seed_summary_detects_changed_calibration_snapshot(self):
        with TemporaryDirectory() as folder:
            paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
            (paths[1]/"calibration.json").write_text(json.dumps({"data_origin":"software_fixture","T_C":2.}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"fixed protocol"):
                analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_seed_summary_rejects_duplicate_seed_in_different_directories(self):
        with TemporaryDirectory() as folder:
            paths=[write_run(Path(folder)/name,0) for name in ("first","second")]
            with self.assertRaisesRegex(ValueError,"duplicate seed 0"):
                analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_seed_summary_allows_consistently_absent_optional_artifacts(self):
        with TemporaryDirectory() as folder:
            paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
            for path in paths:
                file=path/"configuration.json"
                config=json.loads(file.read_text(encoding="utf-8"))
                del config["experiment"]["calibration"];del config["model"]["initial_checkpoint"]
                file.write_text(json.dumps(config),encoding="utf-8")
                file=path/"environment.json"
                environment=json.loads(file.read_text(encoding="utf-8"))
                environment["calibration_sha256"]=None;environment["initial_checkpoint_sha256"]=None
                file.write_text(json.dumps(environment),encoding="utf-8")
                (path/"calibration.json").unlink()
            self.assertEqual(len(analyze_run_dirs(paths,expected_seeds=(0,1))["summary"]),3)
            file=paths[1]/"environment.json"
            environment=json.loads(file.read_text(encoding="utf-8"));environment["initial_checkpoint_sha256"]="c"*64
            file.write_text(json.dumps(environment),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"fixed protocol"):
                analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_remote_seed_summary_binds_worker_data_and_requires_fingerprints(self):
        with TemporaryDirectory() as folder:
            paths=[write_run(Path(folder)/f"seed_{seed}",seed) for seed in (0,1)]
            for path in paths:
                file=path/"configuration.json"
                config=json.loads(file.read_text())
                config["communication"]={"mode":"http","workers":{"client1":{"worker_id":"node1","url":"http://127.0.0.1:8765"}}}
                file.write_text(json.dumps(config))
                file=path/"environment.json"
                environment=json.loads(file.read_text())
                environment["workers"]={"client1":{"worker_id":"node1","partition_sha256":"d"*64,"logical_partition_sha256":"e"*64,"implementation_version":"0.3.0"}}
                file.write_text(json.dumps(environment))
            self.assertEqual(len(analyze_run_dirs(paths,expected_seeds=(0,1))["summary"]),3)
            file=paths[1]/"environment.json"
            environment=json.loads(file.read_text())
            environment["workers"]["client1"]["partition_sha256"]="f"*64
            file.write_text(json.dumps(environment))
            with self.assertRaisesRegex(ValueError,"fixed protocol"):
                analyze_run_dirs(paths,expected_seeds=(0,1))
            del environment["workers"]
            file.write_text(json.dumps(environment))
            with self.assertRaisesRegex(ValueError,"worker identities"):
                analyze_run_dirs(paths,expected_seeds=(0,1))

    def test_case_and_selection_rta_use_distinct_populations_and_missing_misses(self):
        hit={"predicted_class":0,"predicted_box":[0,0,.5,.5],"renderer_class":0,"renderer_box":[0,0,.5,.5]}
        miss={**hit,"predicted_class":1}
        absent={**hit,"predicted_class":None,"predicted_box":None}
        retained=[{"id":"a","rta_observation":hit},{"id":"b","rta_observation":miss},{"id":"a","rta_observation":hit}]
        events=[{"client_id":"1","round_index":1,"case_id":"a",**hit},
                {"client_id":"1","round_index":1,"case_id":"a","epoch":2,**hit},
                {"client_id":"1","round_index":2,"case_id":"a",**absent},
                {"client_id":"2","round_index":1,"case_id":"a",**hit}]
        verification=[{"case_id":"a","entered_verification":True,"retained":True},
                      {"case_id":"b","entered_verification":True,"retained":True},
                      {"case_id":"c","entered_verification":True,"retained":False},
                      {"descriptor_id":"d","reason":"allocated"}]
        result=case_event_proportions(retained,events,verification,qualified_count=2,assessed_count=5)
        self.assertEqual(result["retained_rta"]["numerator"],1)
        self.assertEqual(result["retained_rta"]["denominator"],2)
        self.assertAlmostEqual(result["selected_rta"]["proportion"],2/3)
        self.assertEqual(result["selected_rta"]["denominator"],3)
        self.assertAlmostEqual(result["qualification"]["proportion"],.4)
        self.assertAlmostEqual(result["retention"]["proportion"],2/3)

    def test_rta_empty_none_conflicting_duplicates_and_unknown_reference(self):
        empty=case_event_proportions([],[],[],qualified_count=0,assessed_count=0)
        self.assertIsNone(empty["retained_rta"]["proportion"])
        self.assertIsNone(empty["selected_rta"]["proportion"])
        self.assertIsNone(empty["qualification"]["proportion"])
        self.assertIsNone(empty["retention"]["proportion"])
        good={"id":"x","predicted_class":0,"predicted_box":[0,0,1,1],"renderer_class":0,"renderer_box":[0,0,1,1]}
        with self.assertRaises(ValueError):
            case_event_proportions([good,{**good,"predicted_class":1}])
        with self.assertRaises(ValueError):
            case_event_proportions([{"id":"x","source_class":0}])
        with self.assertRaises(ValueError):
            case_event_proportions([{**good,"agreement":False}])

    def test_qualification_assessed_counts_include_all_dispositions_once(self):
        records=[]
        for uid,disposition in (("a","qualified"),("b","correction"),("c","suppressed")):
            records.append({"client_id":"1","round_index":0,"local_identity":"image","id":uid,"disposition":disposition})
        result=case_event_proportions(candidate_records=records+[deepcopy(records[0])])
        self.assertEqual(result["qualification"]["numerator"],1)
        self.assertEqual(result["qualification"]["denominator"],3)
        self.assertAlmostEqual(result["qualification"]["proportion"],1/3)
        self.assertIsNone(case_event_proportions()["qualification"]["denominator"])


if __name__=="__main__":
    unittest.main()
