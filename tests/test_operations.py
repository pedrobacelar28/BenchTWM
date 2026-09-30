"""CLI, run manifests, queue results, and the summary report."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from benchtwm.cli import main
from benchtwm.operations.artifacts import effective_status, initial_manifest, run_directory, write_manifest
from benchtwm.operations.queue import _write_suite_results
from benchtwm.operations.summary import summarize_runs
from benchtwm.utils.io import save_json


def _cli(capsys, *argv) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_train_dry_run_emits_one_json_plan(capsys, tmp_path):
    code, out, _ = _cli(capsys, "train", "epicare/smart/dev", "dqn", "--artifact-root", str(tmp_path), "--dry-run", "--json")
    plan = json.loads(out)
    assert code == 0 and plan["dataset_id"] == "epicare/smart/dev" and plan["family"] == "model_free"
    assert plan["preflight"]["ready"] is False and plan["model_selection"]["meaning"] == "validation TD loss"


def test_unknown_method_is_a_clean_error(capsys):
    code, _, err = _cli(capsys, "train", "epicare/smart/dev", "ppo", "--dry-run")
    assert code == 2 and "Available:" in err and "Traceback" not in err


def test_ope_rejects_canonical_write_for_a_counterfactual_target(capsys):
    code, _, err = _cli(capsys, "ope", "placeholder", "--target-mode", "stochastic", "--write-canonical")
    assert code == 2 and "--write-canonical is only valid" in err


def test_console_script_entry_points():
    for command in ([sys.executable, "-m", "benchtwm", "catalog", "--json"], [sys.executable, "scripts/bench.py", "catalog", "--json"]):
        payload = json.loads(subprocess.run(command, text=True, capture_output=True, check=True).stdout)
        assert len(payload["datasets"]) == 16


def test_manifest_lifecycle_and_stale_detection(tmp_path):
    params = {"dataset_id": "epicare/smart/dev", "env_id": "epicare", "algo": "dqn", "method_id": "dqn", "seed": 4}
    run_dir = run_directory(params, root=tmp_path)
    assert run_dir.relative_to(tmp_path).parts[:5] == ("runs", "epicare", "smart", "dev", "dqn")
    manifest = write_manifest(run_dir, initial_manifest(params, run_dir=run_dir), status="running")
    assert effective_status(manifest) == "running"  # this process is alive
    manifest["process"]["pid"] = 2**22 + 12345  # no such process
    assert effective_status(manifest) == "stale"
    assert write_manifest(run_dir, manifest, status="interrupted")["finished_at"]


def _fake_run(root, method_id, family, mean_return, seed=0):
    run_dir = root / "runs" / "epicare" / "smart" / "dev" / method_id / f"seed-{seed}"
    save_json({"run_id": f"{method_id}-{seed}", "dataset_id": "epicare/smart/dev", "env_id": "epicare", "method_id": method_id,
               "family": family, "seed": seed, "status": "completed"}, run_dir / "run_manifest.json")
    evaluation = {"mean_return": mean_return, "mean_success": mean_return / 10, "mean_length": 3.0}
    save_json(evaluation, run_dir / "eval_summary.json")
    save_json({"eval_summary": evaluation, "timing_summary": {"optimization_seconds": 1.0}, "ope_summary": {"WPDIS_estimate": 0.5}}, run_dir / "metrics_summary.json")
    return run_dir


def test_summary_computes_planning_gain_against_the_best_model_free_method(tmp_path):
    _fake_run(tmp_path, "dqn", "model_free", 10.0)
    _fake_run(tmp_path, "bc", "model_free", 8.0)
    _fake_run(tmp_path, "raw-model", "model_based", 7.0)
    _fake_run(tmp_path, "raw-model", "model_based", 9.0, seed=1)
    payload = summarize_runs(tmp_path)
    assert payload["best_model_free"] == {"epicare/smart/dev": "dqn"}
    (gain,) = payload["planning_gain"]
    assert gain["method_id"] == "raw-model" and gain["planning_gain_return"] == pytest.approx(-2.0)
    for name in ("runs.csv", "aggregate.csv", "planning_gain.csv", "summary.json", "report.md"):
        assert (tmp_path / "reports" / "summary" / name).exists()


def test_suite_results_use_the_same_summary(tmp_path):
    jobs = []
    for method_id, family, value in (("dqn", "model_free", 10.0), ("raw-model", "model_based", 7.0)):
        run_dir = _fake_run(tmp_path, method_id, family, value)
        jobs.append({"name": method_id, "status": "completed", "output_dir": str(run_dir), "method_id": method_id, "dataset_id": "epicare/smart/dev"})
    queue_dir = tmp_path / "queues" / "q"
    queue_dir.mkdir(parents=True)
    payload = json.loads(_write_suite_results(queue_dir, {"queue_id": "q", "finished_at": "now", "failures": 0, "jobs": jobs}).read_text())
    assert payload["suite"]["status"] == "completed" and payload["counts"]["completed"] == 2
    assert payload["best_model_free"] == {"epicare/smart/dev": "dqn"}
    assert payload["planning_gain"][0]["planning_gain_return"] == pytest.approx(-3.0)
