from optimagic.optimization.process_results import _sum_or_none


def test_sum_or_none():
    assert _sum_or_none([1, 2, 3]) == 6
    assert _sum_or_none([1, 2, None]) is None
