from mdsim import MDSim


def test_describe():
    m = MDSim()
    assert "MDSim" in m.describe()
