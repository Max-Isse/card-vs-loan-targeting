"""Checks against known answers. Run:  python tests/test_core.py   (or pytest, if installed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import decompose, policy, simulate  # noqa: E402


def test_decomposition_identity_is_exact():
    rng = np.random.default_rng(0)
    for _ in range(50):
        cl = np.array([rng.integers(100, 1000), rng.integers(10, 400), 0, 0])
        cb = rng.integers(1, 500, 4)
        d = decompose.decomposition(cl, cb)
        assert abs(d["card_takeup_both_arm"] - (d["new"] + d["diverted"] + d["took_both"])) < 1e-12


def test_no_cannibalisation_when_choices_are_independent():
    pop = simulate.population(10_000, np.random.default_rng(1))
    PL = simulate.choice_probs(pop, "loan")
    PC = simulate.choice_probs(pop, "card")
    PB = simulate.choice_probs(pop, "both", kappa=0.0, attention=0.0)
    assert np.allclose(PB[:, 1] + PB[:, 3], PL[:, 1])  # loan take-up unchanged
    assert np.allclose(PB[:, 2] + PB[:, 3], PC[:, 2])  # card take-up unchanged


def test_policy_value_of_single_arm_is_that_arm_mean():
    d = simulate.simulate_experiment(seed=2, n=3_000)
    for a in range(3):
        v = policy.policy_value(d["y"], d["arm"], np.full(len(d["y"]), a))
        assert abs(v - d["y"][d["arm"] == a].mean()) < 1e-9


def test_policy_value_is_unbiased_for_a_targeted_policy():
    d = simulate.simulate_experiment(seed=3, n=600_000)
    mu = np.column_stack([simulate.expected_value(d["pop"], a) for a in simulate.ARMS])
    choice = mu.argmax(1)
    est = policy.policy_value(d["y"], d["arm"], choice)
    assert abs(est - mu.max(1).mean()) < 25  # about 3 standard errors


def test_eb_shrinks_fully_when_there_is_no_real_variation():
    rng = np.random.default_rng(4)
    se2 = np.full(30, 100.0**2)
    d = 50 + rng.normal(0, 100, 30) * 0.5  # spread smaller than the noise
    shrunk, tau2 = policy.eb_shrink(d, se2)
    assert tau2 == 0.0 and np.allclose(shrunk, shrunk[0])


def test_regression_recovers_noiseless_truth():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.normal(0, 1, 3_000), rng.normal(0, 0.7, 3_000), rng.beta(2, 3, 3_000), rng.normal(0, 1, 3_000)])
    arm = rng.integers(0, 3, 3_000)
    coef = rng.normal(0, 100, (3, policy.basis(X).shape[1]))
    y = (policy.basis(X) * coef[arm]).sum(1)
    pred = policy.regression_predict(y, arm, X, X, ridge=0.0)
    assert np.allclose(pred, policy.basis(X) @ coef.T, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
