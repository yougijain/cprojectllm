from scripts.eval_gate import decide


def test_gate_allows_two_points_and_no_more():
    base = {"field_accuracy": 0.90}
    assert decide(base, {"field_accuracy": 0.90})[0]
    assert decide(base, {"field_accuracy": 0.95})[0]
    assert decide(base, {"field_accuracy": 0.88})[0]  # exactly 2 points
    ok, msg = decide(base, {"field_accuracy": 0.879})
    assert not ok and "FAILS" in msg
