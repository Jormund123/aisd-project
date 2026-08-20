"""
Known-answer unit tests for src/pairwise.py (Approach 3 math).

No LLM, no files, fully deterministic. Run from the repo root:
    ./venv/bin/python -m pytest tests/approach3/test_pairwise.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "approach3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pairwise import (  # noqa: E402
    win_count_scores,
    bradley_terry,
    pbo_acquisition,
    pairwise_accuracy,
    ranking_metrics,
)


def _total_order_duels(order):
    """Every pair compared once, with the earlier element in `order` always winning."""
    duels = []
    for a_pos in range(len(order)):
        for b_pos in range(a_pos + 1, len(order)):
            i, j = order[a_pos], order[b_pos]
            duels.append((i, j, i))  # i (ranked higher) beats j
    return duels


def test_win_count_basic():
    # A>B>C>D total order: A wins 3, B wins 2, C wins 1, D wins 0.
    duels = _total_order_duels([0, 1, 2, 3])
    scores = win_count_scores(duels, 4)
    assert list(scores) == [3.0, 2.0, 1.0, 0.0]


def test_win_count_empty():
    assert list(win_count_scores([], 3)) == [0.0, 0.0, 0.0]


def test_bradley_terry_total_order_is_monotone():
    duels = _total_order_duels([0, 1, 2, 3])
    u, se = bradley_terry(duels, 4)
    # utility strictly decreasing A..D, and centered
    assert u[0] > u[1] > u[2] > u[3]
    assert abs(u.mean()) < 1e-6
    assert np.all(se > 0)


def test_bradley_terry_agrees_with_win_count_ordering():
    duels = _total_order_duels([2, 0, 3, 1])  # ranking 2>0>3>1
    u, _ = bradley_terry(duels, 4)
    wc = win_count_scores(duels, 4)
    assert np.argsort(-u).tolist() == np.argsort(-wc).tolist()


def test_bradley_terry_symmetry():
    # One win each way between 0 and 1 -> near-equal utilities.
    duels = [(0, 1, 0), (0, 1, 1)]
    u, _ = bradley_terry(duels, 2)
    assert abs(u[0] - u[1]) < 1e-6


def test_bradley_terry_empty_is_wide_and_zero():
    u, se = bradley_terry([], 5)
    assert np.allclose(u, 0.0)
    assert np.all(se > 1.0)  # no information -> wide uncertainty


def test_pbo_skips_evaluated():
    u = np.array([3.0, 2.0, 1.0])
    se = np.zeros(3)
    mask = np.array([True, False, False])  # best one already evaluated
    assert pbo_acquisition(u, se, mask, kappa=0.0) == 1


def test_pbo_kappa_zero_is_greedy():
    u = np.array([1.0, 5.0, 2.0])
    se = np.array([9.0, 0.0, 0.0])
    mask = np.array([False, False, False])
    assert pbo_acquisition(u, se, mask, kappa=0.0) == 1  # highest utility


def test_pbo_kappa_rewards_uncertainty():
    u = np.array([1.0, 5.0, 2.0])
    se = np.array([9.0, 0.0, 0.0])
    mask = np.array([False, False, False])
    # large kappa -> candidate 0 (huge uncertainty) wins over 1 (high utility)
    assert pbo_acquisition(u, se, mask, kappa=2.0) == 0


def test_pbo_all_evaluated_returns_none():
    assert pbo_acquisition(np.zeros(2), np.zeros(2), np.array([True, True])) is None


def test_pairwise_accuracy_perfect_and_reversed():
    pred = np.array([3.0, 2.0, 1.0])
    true = np.array([30.0, 20.0, 10.0])
    assert pairwise_accuracy(pred, true) == 1.0
    assert pairwise_accuracy(pred, -true) == 0.0


def test_ranking_metrics_perfect():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    true = np.array([10.0, 20.0, 30.0, 40.0])
    m = ranking_metrics(pred, true)
    assert abs(m["spearman"] - 1.0) < 1e-9
    assert abs(m["kendall"] - 1.0) < 1e-9
    assert m["pairwise_accuracy"] == 1.0


def test_ranking_metrics_reversed():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    true = np.array([40.0, 30.0, 20.0, 10.0])
    m = ranking_metrics(pred, true)
    assert abs(m["spearman"] + 1.0) < 1e-9
    assert m["pairwise_accuracy"] == 0.0


if __name__ == "__main__":
    # Plain runner so the suite works without pytest installed.
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok   {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
