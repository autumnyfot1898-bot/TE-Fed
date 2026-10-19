"""Check registered methods with generated test inputs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys
import time
import traceback


VIEWS = {"participating", "unseen", "new-client", "hard"}
PSEUDO_METHODS = {"confidence", "consistent_teacher_fedawa", "fedsto", "fsslod"}
TEACHER_METHODS = PSEUDO_METHODS - {"confidence"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value, path="value"):
    """Allow undefined class AP values, but reject any NaN or infinity."""
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, bool):
        require(math.isfinite(value), f"Nonfinite number: {path}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_methods(values, registry):
    methods = list(registry) if not values else list(values)
    require(bool(methods), "No methods selected")
    require(len(set(methods)) == len(methods), "Method names must be distinct")
    require(set(methods) <= set(registry), f"Unknown methods: {sorted(set(methods) - set(registry))}")
    return methods


def tiny_architecture():
    """Retain native three-level end2end Detect and native loss with a small trunk."""
    return {"nc": 2, "end2end": True, "reg_max": 1,
            "backbone": [[-1, 1, "Conv", [width, 3, 2]] for width in (8, 16, 24, 32, 48)],
            "head": [[[2, 3, 4], 1, "Detect", ["nc"]]]}


def validate_metrics(metrics):
    require(set(metrics) == VIEWS, "Evaluation must contain all four views")
    finite_tree(metrics, "metrics")
    for view, values in metrics.items():
        for key in ("mAP50", "mAP50_95", "AP75"):
            value = values.get(key)
            require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value),
                    f"Missing or nonfinite {view}.{key}")
            require(0 <= value <= 1, f"AP outside [0, 1]: {view}.{key}")
        require(values.get("num_images") == 2, f"Incorrect generated view cardinality: {view}")


def validate_rounds(rows, method, policies, clients, rounds, epochs):
    require([row.get("round_index") for row in rows] == list(range(rounds)), "Round history is incomplete")
    finite_tree(rows, "rounds")
    observations = []
    for row in rows:
        round_index = row["round_index"]
        expected = policies[round_index * epochs:(round_index + 1) * epochs]
        require(row.get("baseline_policy") == expected[0], f"Round policy mismatch: {method}/{round_index}")
        require(set(row.get("clients", {})) == set(clients), "Client history is incomplete")
        aggregation = row.get("aggregation")
        if method != "local":
            require(isinstance(aggregation, dict) and aggregation.get("baseline_policy") == expected[0],
                    f"Missing actual aggregation policy: {method}/{round_index}")
        else:
            require(aggregation is None, "Local method unexpectedly aggregated")
        for client, local in row["clients"].items():
            require(local.get("baseline_policies") == expected, f"Client policy mismatch: {method}/{client}/{round_index}")
            require(local.get("assessment_policy") == expected[0], "Assessment policy does not match training")
            attempted, successful, skipped = (local.get(key) for key in
                ("attempted_optimizer_steps", "optimizer_steps", "amp_skipped_steps"))
            require(all(type(value) is int and value >= 0 for value in (attempted, successful, skipped)),
                    "Missing optimizer counters")
            require(successful > 0 and attempted == successful + skipped, "Optimizer counters are inconsistent")
            require(isinstance(local.get("mean_step_loss"), Real), "Missing mean step loss")
            require(local.get("presentations", {}).get("labeled") == 2 * epochs,
                    "Generated labeled observations were not all trained")
            if method in PSEUDO_METHODS:
                require(local["presentations"].get("pseudo", 0) > 0, f"Pseudo path was not exercised: {method}")
            if method in TEACHER_METHODS:
                require(local.get("consistency_pairs", 0) > 0, f"Consistency path was not exercised: {method}")
                require(local.get("teacher_updates") == successful, "Client teacher updates differ from successful steps")
            if method == "fedsto":
                stages = local.get("stage_history", [])
                require([stage.get("epoch_index") for stage in stages] == list(range(epochs)), "FedSTO stage history is incomplete")
                require([stage.get("freeze_trunk") for stage in stages] == [policy["freeze_trunk"] for policy in expected],
                        "Actual FedSTO stages differ from policy")
            observations.append({"round_index": round_index, "client_id": client,
                "attempted_optimizer_steps": attempted, "successful_optimizer_steps": successful,
                "amp_skipped_steps": skipped, "mean_step_loss": local["mean_step_loss"],
                "presentations": local["presentations"], "consistency_pairs": local.get("consistency_pairs", 0),
                "candidate_counts": local.get("candidate_counts"), "teacher_updates": local.get("teacher_updates"),
                "stage_history": local.get("stage_history"), "baseline_policies": local["baseline_policies"]})
    return observations


def validate_summary(summary):
    require(summary.get("data_origin") == "generated", "Summary must identify generated data")
    requested = summary.get("requested_methods", [])
    require(requested and len(set(requested)) == len(requested), "Requested method coverage is invalid")
    rows = summary.get("methods", [])
    require([row.get("method") for row in rows] == requested, "Summary method coverage differs from request")
    require(all(row.get("status") == "passed" for row in rows), "One or more method checks failed")
    require(summary.get("paper_reproduction") is False, "Generated checks must not claim paper reproduction")


@contextmanager
def observe_baseline_calls():
    """Count real helper calls without changing their return values or arguments."""
    from tefed import baselines
    names = ("update_ema_teacher", "fedsto_set_trainable", "proximal_penalty", "elastic_penalty", "orthogonality_penalty")
    originals = {name: getattr(baselines, name) for name in names}
    calls = {name: [] for name in names}

    def wrap(name, function):
        def observed(*args, **kwargs):
            result = function(*args, **kwargs)
            detail = {}
            if name == "update_ema_teacher":
                detail["decay"] = float(args[2] if len(args) > 2 else kwargs["decay"])
            elif name == "fedsto_set_trainable":
                detail = deepcopy(result)
            else:
                detail["value"] = float(result.detach())
            calls[name].append(detail)
            return result
        return observed

    try:
        for name, function in originals.items():
            setattr(baselines, name, wrap(name, function))
        yield calls
    finally:
        for name, function in originals.items():
            setattr(baselines, name, function)


def verify_run(result, method, requested_config, evidence, observed_calls):
    import torch
    from tefed.baselines import policy_schedule
    from tefed.data import load_manifest
    from tefed.engine import evaluate_model, load_checkpoint
    from tefed.resume import load_snapshot

    output = Path(result["output_dir"])
    required = ["configuration.json", "calibration.json", "environment.json", "rounds.jsonl", "metrics.json",
                "final.pt", "final.pool.json", "resume.pt", "source_states.json", "status.json",
                "predictions/evaluation_manifest.json"]
    required += [f"verification_round_{index:03d}.json" for index in range(requested_config["experiment"]["rounds"])]
    for name in required:
        require((output / name).is_file(), f"Missing output: {name}")
    saved = read_json(output / "configuration.json")
    require(saved["experiment"]["method"] == method, "Saved configuration method mismatch")
    require(saved["experiment"].get("test_mode") is True, "Generated run lacks test mode")
    policies = policy_schedule(method, saved)
    require(saved.get("resolved_baseline_policies") == policies, "Saved resolved policy schedule mismatch")
    for section in ("experiment", "baselines", "optimizer", "loss", "decision", "evaluation"):
        require(saved.get(section) == requested_config.get(section), f"Requested/saved configuration mismatch: {section}")
    manifest = load_manifest(saved["experiment"]["manifest"], strict_files=True)
    clients = sorted({str(record.client_id) for record in manifest.by_split("labeled")})
    require(manifest.metadata.get("data_origin") == "generated", "Fixture origin is not generated")
    rows = read_jsonl(output / "rounds.jsonl")
    require(rows == result["rounds"], "Returned and persisted round histories differ")
    client_steps = validate_rounds(rows, method, policies, clients,
                                  saved["experiment"]["rounds"], saved["experiment"]["local_epochs"])
    metrics = read_json(output / "metrics.json")
    validate_metrics(metrics)
    require(metrics == result["metrics"], "Returned and persisted metrics differ")
    require(read_json(output / "status.json").get("status") == "completed", "Run status is not completed")
    snapshot = load_snapshot(output / "resume.pt")
    require(snapshot.get("phase") == "completed", "Resume snapshot is incomplete")
    binding = snapshot["binding"]
    source_root = Path(__file__).resolve().parents[1] / "src/tefed"
    for name, expected_hash in binding["source_files"].items():
        require(file_sha256(source_root / name) == expected_hash, f"Source changed during check: {name}")
    for path, expected_hash in binding["input_files"].items():
        require(file_sha256(path) == expected_hash, f"Bound input changed: {path}")
    write_json(evidence / "input_configuration_binding.json", binding)
    aggregation_state = snapshot.get("aggregation_state")
    if policies[0]["aggregation"] == "fedawa_paper":
        require(isinstance(aggregation_state, dict) and aggregation_state.get("client_ids") == clients,
                "Missing persisted FedAWA client ordering")
        require(aggregation_state.get("logits") == rows[-1]["aggregation"].get("logits"),
                "FedAWA persisted logits differ from final aggregation")
        finite_tree(aggregation_state, "aggregation_state")
        for row in rows:
            actual = row["aggregation"]
            require(actual.get("definition") == "fedawa_paper_eq3", "FedAWA objective definition mismatch")
            require(actual.get("probability_timing") == "after_final_optimizer_step", "FedAWA weights were not read after optimization")
            require(len(actual.get("logits", [])) == len(clients), "Missing FedAWA logits")
            expected_weights = torch.tensor(actual["logits"], dtype=torch.float64).softmax(0).tolist()
            require(all(abs(actual["weights"][client] - weight) <= 1e-12 for client, weight in zip(clients, expected_weights)),
                    "FedAWA aggregation weights differ from persisted logits")
        write_json(evidence / "persisted_aggregation_state.json", aggregation_state)

    model, router, checkpoint = load_checkpoint(result["checkpoint"], device=saved["experiment"]["device"])
    require(len(model.heads) == (6 if policies[0]["use_experts"] else 1), "Checkpoint detector heads differ from method policy")
    require(checkpoint["config"] == {key: value for key, value in saved.items() if key != "resolved_baseline_policies"},
            "Checkpoint configuration mismatch")
    require(checkpoint["config"]["experiment"]["method"] == method, "Checkpoint method mismatch")
    require(checkpoint.get("completed_rounds") == saved["experiment"]["rounds"], "Checkpoint round count mismatch")
    for module in (model, router):
        for key, value in module.state_dict().items():
            require(not value.is_floating_point() or bool(torch.isfinite(value).all()), f"Nonfinite checkpoint tensor: {key}")
    persistent = snapshot.get("client_runtime", {})
    teacher_clients = [client for client in clients if persistent.get(client, {}).get("teacher") is not None]
    if method in TEACHER_METHODS:
        require(teacher_clients == clients, "Missing persisted teacher state")
        require(len(observed_calls["update_ema_teacher"]) == sum(row["successful_optimizer_steps"] for row in client_steps),
                "Teacher updates do not match successful optimizer steps")
    else:
        require(not teacher_clients and not observed_calls["update_ema_teacher"], "Unexpected teacher state or updates")
    if method == "fedsto":
        require(len(observed_calls["fedsto_set_trainable"]) == len(clients) * len(policies), "Missing FedSTO stages")
        require({policy["freeze_trunk"] for policy in policies} == {True, False}, "Both FedSTO stages did not execute")
    for name, active in (("proximal_penalty", method == "fedprox"), ("elastic_penalty", method == "fsslod"),
                         ("orthogonality_penalty", method == "fedsto")):
        require(bool(observed_calls[name]) == active, f"Unexpected penalty call coverage: {name}")
    finite_tree(observed_calls, "observed_calls")
    write_json(evidence / "observed_baseline_calls.json", observed_calls)

    calibration = read_json(output / "calibration.json")
    require(calibration.get("data_origin") == "generated", "Calibration origin is not generated")
    kwargs = {}
    if method == "local":
        require(checkpoint.get("checkpoint_kind") == "local_bundle", "Missing local checkpoint bundle")
        require(set(checkpoint.get("client_models", {})) == set(clients), "Missing local client models")
        kwargs = {"client_models": checkpoint["client_models"], "local_new_client_id": checkpoint["local_new_client_id"]}
    elif method == "fedbn":
        require(checkpoint.get("checkpoint_kind") == "fedbn_bundle", "Missing FedBN checkpoint bundle")
        require(set(checkpoint.get("client_bn", {})) == set(clients), "Missing client BN states")
        kwargs = {"client_bn": checkpoint["client_bn"]}
    reloaded = evaluate_model(model, router, manifest, checkpoint["states"], calibration,
                              checkpoint["config"], evidence / "reloaded_predictions", **kwargs)
    validate_metrics(reloaded)
    require(reloaded == metrics, "Saved/reloaded checkpoint metrics differ")
    for view in sorted(VIEWS):
        original = output / "predictions" / f"{view}.jsonl"
        reproduced = evidence / "reloaded_predictions" / f"{view}.jsonl"
        predictions = read_jsonl(original)
        finite_tree(predictions, f"predictions.{view}")
        require(len(predictions) == 2 and len({row["id"] for row in predictions}) == 2,
                f"Missing or duplicate predictions: {view}")
        require(predictions == read_jsonl(reproduced), f"Reloaded predictions differ: {view}")
    write_json(evidence / "reloaded_metrics.json", reloaded)
    return {"status": "passed", "method": method, "output_dir": str(output), "metrics": metrics,
            "capacity": read_json(output / "environment.json")["capacity"],
            "checkpoint_sha256": file_sha256(result["checkpoint"]), "checkpoint_kind": checkpoint.get("checkpoint_kind", "global"),
            "input_configuration_binding": str(evidence / "input_configuration_binding.json"),
            "resolved_baseline_policies": policies, "client_optimizer_steps": client_steps,
            "teacher_clients": teacher_clients, "observed_call_counts": {key: len(value) for key, value in observed_calls.items()},
            "persisted_aggregation_state": aggregation_state,
            "aggregation": [{"round_index": row["round_index"], "evidence": row.get("aggregation")} for row in rows],
            "saved_reloaded_metrics_equal": True, "saved_reloaded_predictions_equal": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for generated inputs and evidence")
    parser.add_argument("--methods", nargs="+", help="Registered names; omitted means every registered method")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fedawa-mode", choices=("directional", "fedawa_paper"), default="directional",
                        help="Optional CT+FedAWA aggregation mode for an additional software check")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.threads < 1:
        parser.error("--seed must be nonnegative and --threads positive")
    from tefed.baselines import METHOD_REGISTRY, policy_schedule
    try:
        methods = parse_methods(args.methods, METHOD_REGISTRY)
    except ValueError as exc:
        parser.error(str(exc))
    import torch
    from tefed.engine import run_experiment
    from tefed.software_fixture import create_fixture

    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    base = create_fixture(root / "generated_inputs")
    base["experiment"].update(seed=args.seed, local_epochs=2, device=args.device)
    base["model"]["architecture"] = tiny_architecture()
    default_policies = {method: policy_schedule(method, base) for method in methods}
    base["baselines"] = {method: {"pseudo_threshold": 0.0} for method in sorted(PSEUDO_METHODS)}
    if args.fedawa_mode == "fedawa_paper":
        require("consistent_teacher_fedawa" in methods, "--fedawa-mode requires consistent_teacher_fedawa")
        base["baselines"]["consistent_teacher_fedawa"]["aggregation_mode"] = args.fedawa_mode
    summary = {"schema_version": 1, "data_origin": "generated", "paper_reproduction": False,
        "purpose": "Native detector software checks with generated inputs and untrained initialization",
        "coverage_overrides": base["baselines"], "default_baseline_policies": default_policies,
        "limitations": ["Tiny native detector replaces the paper architecture", "No pretrained or real dataset accuracy claim",
                        "Rendering and correction reuse are disabled; only correction-capable method dispatch is checked",
                        "Expert contrastive and consistency coefficients remain disabled in this fixture"],
        "python_executable": sys.executable, "threads": args.threads, "seed": args.seed,
        "runner_sha256": file_sha256(__file__),
        "requested_methods": methods, "registered_methods": list(METHOD_REGISTRY),
        "all_registered_methods_requested": set(methods) == set(METHOD_REGISTRY),
        "rounds": 2, "local_epochs": 2, "started_utc": datetime.now(timezone.utc).isoformat(), "methods": []}
    write_json(root / "check_configuration.json", base)
    write_json(root / "summary.json", summary)
    started = time.perf_counter()
    for method in methods:
        config = deepcopy(base)
        output = root / "runs" / method
        config["experiment"].update(method=method, output=str(output))
        evidence = root / "evidence" / method
        evidence.mkdir(parents=True)
        write_json(evidence / "requested_configuration.json", config)
        method_started = time.perf_counter()
        try:
            with observe_baseline_calls() as calls:
                result = run_experiment(config, seed=args.seed, output_dir=output)
            row = verify_run(result, method, config, evidence, calls)
        except Exception as exc:
            (evidence / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            row = {"method": method, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "failure_details": str(evidence / "failure.txt"), "output_dir": str(output)}
        row["seconds"] = time.perf_counter() - method_started
        summary["methods"].append(row)
        write_json(evidence / "check_result.json", row)
        write_json(root / "summary.json", summary)
        print(f"baseline check {method}: {row['status']} ({row['seconds']:.2f}s)", flush=True)
    try:
        validate_summary(summary)
        summary["status"] = "passed"
    except ValueError as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
    summary["seconds"] = time.perf_counter() - started
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "summary.json", summary)
    print(f"Generated baseline checks: {summary['status']}; evidence: {root / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
