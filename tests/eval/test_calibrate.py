from docparse.eval.calibrate import PageFeatures, choose, sweep


def _f(i: int, rate: float, rec: float, prec: float) -> PageFeatures:
    return PageFeatures(f"p{i}", "multi_column", rate, rec, prec, 0.5, False)


def test_sweep_and_choose_respect_flag_budget() -> None:
    feats = [_f(i, 1.0, 0.99, 0.99) for i in range(95)]
    feats += [_f(100 + i, 0.0, 0.5, 0.9) for i in range(3)]  # bad pages, low recall
    feats += [_f(200 + i, 1.0, 0.6, 0.6) for i in range(2)]  # fine pages, low agreement
    df = sweep(feats, [0.5, 0.7, 0.9], [0.5, 0.7, 0.9])
    row = df.filter((df["recall_min"] == 0.7) & (df["precision_min"] == 0.5)).row(0, named=True)
    assert row["failing_caught_pct"] == 100.0
    assert row["flagged_pct"] == 4.0 + 1.0  # 3 bad + 2 fine flagged by recall 0.6 < 0.7 -> 5/100
    best = choose(df, max_flagged_pct=5.0)
    assert best["flagged_pct"] <= 5.0 and best["failing_caught_pct"] == 100.0


def test_scans_are_never_flagged() -> None:
    f = PageFeatures("s", "old_scans", 0.0, None, None, 0.5, False)
    df = sweep([f], [0.9], [0.9])
    assert df.row(0, named=True)["flagged_pct"] == 0.0
