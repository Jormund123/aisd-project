"""
Known-answer unit tests for src/metrics.py (shared cross-approach metrics).

No LLM, no files, fully deterministic. Run from the repo root:
    ./venv/bin/python tests/test_metrics.py
(or with pytest if installed).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (  # noqa: E402
    best_found,
    simple_regret,
    normalized_score,
    improvement_over_seed,
    first_improvement_iter,
    iters_to_ceiling,
    success_rate,
    regression_metrics,
    sign_accuracy,
    top1_hit,
    ndcg_at_k,
    hit_rate,
)


def test_best_found_includes_start():
    assert best_found([0.1, 0.3, 0.3], 0.5) == 0.5   # start beats trajectory
    assert best_found([0.1, 0.6, 0.6], 0.2) == 0.6


def test_simple_regret():
    assert abs(simple_regret(0.8, 1.0) - 0.2) < 1e-9
    assert np.isnan(simple_regret(0.8, None))


def test_normalized_score():
    # start 0, ceiling 1, found 0.5 -> 0.5
    assert abs(normalized_score(0.5, 0.0, 1.0) - 0.5) < 1e-12
    assert normalized_score(1.0, 0.0, 1.0) == 1.0
    assert np.isnan(normalized_score(0.5, 0.2, 0.2))   # no headroom


def test_improvement_over_seed():
    assert abs(improvement_over_seed(0.6, 0.2) - 0.4) < 1e-12


def test_first_improvement_iter():
    # best-so-far stays at seed for two evals, improves on the third
    assert first_improvement_iter([0.2, 0.2, 0.5, 0.5], 0.2) == 3
    assert first_improvement_iter([0.1, 0.1], 0.2) is None  # never beat seed
    # honors explicit iteration labels
    assert first_improvement_iter([0.2, 0.9], 0.2, iterations=[5, 6]) == 6


def test_iters_to_ceiling():
    assert iters_to_ceiling([0.2, 0.5, 1.0, 1.0], 1.0) == 3
    assert iters_to_ceiling([0.2, 0.5], 1.0) is None
    assert iters_to_ceiling([0.2, 0.5], None) is None


def test_success_rate():
    # vs start 0.2: 0.3 improves, 0.3 no, 0.7 improves, 0.7 no -> 2/4
    assert success_rate([0.3, 0.3, 0.7, 0.7], 0.2) == 0.5
    assert success_rate([0.1, 0.1], 0.2) == 0.0       # nothing beats seed
    assert success_rate([0.3, 0.4, 0.5], 0.2) == 1.0  # every eval a new best


def test_regression_metrics_perfect():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["bias"] == 0.0
    assert abs(m["pearson"] - 1.0) < 1e-9
    assert abs(m["r2"] - 1.0) < 1e-9


def test_regression_metrics_known_bias():
    # pred is true + 1 everywhere: mae=1, rmse=1, bias=+1, pearson=1, r2=1-(3/SS_tot)
    pred = [2.0, 3.0, 4.0]
    true = [1.0, 2.0, 3.0]
    m = regression_metrics(pred, true)
    assert abs(m["mae"] - 1.0) < 1e-9
    assert abs(m["rmse"] - 1.0) < 1e-9
    assert abs(m["bias"] - 1.0) < 1e-9
    assert abs(m["pearson"] - 1.0) < 1e-9
    ss_tot = 2.0  # sum((true-mean)^2) for [1,2,3] = 1+0+1
    assert abs(m["r2"] - (1.0 - 3.0 / ss_tot)) < 1e-9


def test_sign_accuracy():
    # threshold 0: signs of pred vs true. pred [+,+,-], true [+,-,-] -> 2/3 agree
    assert abs(sign_accuracy([1, 1, -1], [1, -1, -1], 0.0) - 2 / 3) < 1e-9
    assert sign_accuracy([1, 2], [5, 6], 0.0) == 1.0


def test_top1_hit():
    assert top1_hit([0.1, 0.9, 0.3], [10, 99, 30]) == 1.0   # both pick index 1
    assert top1_hit([0.9, 0.1, 0.3], [10, 99, 30]) == 0.0   # pred picks 0, true 1


def test_ndcg_perfect_and_reversed():
    pred = [3.0, 2.0, 1.0]
    true = [30.0, 20.0, 10.0]
    assert abs(ndcg_at_k(pred, true, k=3) - 1.0) < 1e-9
    # reversed prediction scores below ideal
    assert ndcg_at_k([1.0, 2.0, 3.0], true, k=3) < 1.0


def test_hit_rate():
    # vs seed 0.2: values 0.5,0.1,0.3 -> 2/3 beat it
    assert abs(hit_rate([0.5, 0.1, 0.3], 0.2) - 2 / 3) < 1e-9
    assert hit_rate([0.0, 0.1], 0.2) == 0.0


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
