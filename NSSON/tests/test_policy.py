from camnsson.policy.offload import decide_target
def test_policy():
    assert decide_target({"target":"jetson"}) == "jetson"
