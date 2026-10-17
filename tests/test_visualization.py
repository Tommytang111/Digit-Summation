import visualization as viz


def test_metric_summary_plot_is_written_and_tolerates_missing_macro_f1(tmp_path):
    main = {"accuracy": 0.95, "macro_f1": 0.78, "mae": 0.11}
    baseline = {"answer": "45", "accuracy": 0.04, "macro_f1": 0.001, "mae": 7.3}
    path = viz.plot_metric_summary(main, baseline, "test", tmp_path / "summary.png", "test summary")
    assert path.exists() and path.stat().st_size > 0
    # Evaluations from before macro F1 existed: that panel is skipped instead of failing.
    old = viz.plot_metric_summary({"accuracy": 0.9, "mae": 0.2}, baseline, "val", tmp_path / "old.png", "old summary")
    assert old.exists()
