from config import ExperimentConfig
from experiment_logging import flatten_metrics, log_evaluation


def test_flatten_metrics_keys_groups_and_skips_non_scalars():
    metrics = {
        "accuracy": 0.95,
        "n": 20,
        "by_image_count": [{"group": 0, "n": 2, "accuracy": 1.0}, {"group": 10, "n": 0, "accuracy": None}],
        "by_sum_range": [{"group": "40-49", "n": 8, "accuracy": 0.5}],
        "majority_baseline": {"rule": "always 45", "answer": "45", "accuracy": 0.04},
        "stress_cases": {"all_0_text": {"accuracy": 0.0, "target": 0, "predictions_seen": [12, 14]}},
        "empty": [],
    }
    assert flatten_metrics(metrics, prefix="test") == {
        "test/accuracy": 0.95,
        "test/n": 20,
        "test/by_image_count/0/n": 2,
        "test/by_image_count/0/accuracy": 1.0,
        "test/by_image_count/10/n": 0,
        "test/by_sum_range/40-49/n": 8,
        "test/by_sum_range/40-49/accuracy": 0.5,
        "test/majority_baseline/accuracy": 0.04,
        "test/stress_cases/all_0_text/accuracy": 0.0,
        "test/stress_cases/all_0_text/target": 0,
    }


def test_log_evaluation_is_a_no_op_when_disabled(tmp_path):
    logging_cfg = ExperimentConfig().logging
    assert logging_cfg.wandb_mode == "disabled"
    log_evaluation(logging_cfg, {}, {"test/accuracy": 1.0}, {}, name="run-eval", group="run", directory=tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing written, wandb never started
