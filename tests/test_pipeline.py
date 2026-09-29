"""End to end on synthetic data: prepare -> baseline -> compare, as subprocesses."""

import json
import os
import subprocess
import sys


def run(module, root, *args):
    env = {**os.environ, "TENDER_SCOUT_ROOT": str(root)}
    subprocess.run([sys.executable, "-m", module, *args], env=env, check=True)


def test_prepare_baseline_compare(project):
    run("tender_scout.prepare", project)
    log = json.loads((project / "reports" / "cleaning_log.json").read_text())
    assert log["lots"] == 1203
    assert log["rows_per_split"] == {"train": 401, "val": 401, "test": 401}
    assert log["cigs_without_prevalent_cpv"] == 12

    run("tender_scout.baseline", project)
    metrics = json.loads((project / "reports" / "baseline_metrics.json").read_text())
    assert [t["C"] for t in metrics["tuning"]] == [0.1, 0.3, 1.0]
    assert metrics["test"]["macro_f1"] > 0.9
    assert metrics["majority_test"]["macro_f1"] < 0.2
    assert (project / "models" / "baseline.joblib").exists()

    run("tender_scout.compare", project)
    assert "TF-IDF + linear SVM" in (project / "reports" / "comparison.md").read_text()
