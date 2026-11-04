from camnsson.fusion.fuser import fuse_scores
def test_fuse():
    assert fuse_scores(0.5, None, 0.5) > 0
