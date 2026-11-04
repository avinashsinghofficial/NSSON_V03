from controllers.decision_engine import choose_target
def test_decision():
    d = choose_target(0.3,0.2,10000,0.5,{"link_delay_ms":10},{})
    assert "target" in d
