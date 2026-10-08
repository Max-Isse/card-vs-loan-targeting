"""Does the card win extra customers or take them from the loan?

Compare the "both" arm with the "loan only" arm (the status quo). Every card taker in
the both arm falls into one of three groups, and the identity is exact:

    card take-up (both arm) = new customers      any product (both) - loan (loan only)
                            + diverted from loan loan (loan only)  - loan (both)
                            + took both          share taking loan and card (both)

Individual customers cannot be labelled new or diverted, but each group's share is a
difference between randomised arms, so it is identified on average.
Uncertainty comes from a multinomial bootstrap of each arm's outcome counts.
"""
import numpy as np

# outcome codes: 0 none, 1 loan, 2 card, 3 both


def shares(counts):
    """Take-up shares from (..., 4) outcome counts."""
    c = np.asarray(counts, float)
    n = c.sum(-1)
    return {
        "loan": (c[..., 1] + c[..., 3]) / n,
        "card": (c[..., 2] + c[..., 3]) / n,
        "both": c[..., 3] / n,
        "any": 1 - c[..., 0] / n,
    }


def decomposition(counts_loan_arm, counts_both_arm):
    L, Bo = shares(counts_loan_arm), shares(counts_both_arm)
    return {
        "loan_takeup_loan_arm": L["loan"],
        "loan_takeup_both_arm": Bo["loan"],
        "card_takeup_both_arm": Bo["card"],
        "new": Bo["any"] - L["any"],
        "diverted": L["loan"] - Bo["loan"],
        "took_both": Bo["both"],
    }


def counts(outcome, mask):
    return np.bincount(outcome[mask], minlength=4)


def bootstrap(counts_loan_arm, counts_both_arm, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    cl, cb = np.asarray(counts_loan_arm), np.asarray(counts_both_arm)
    bl = rng.multinomial(cl.sum(), cl / cl.sum(), B)
    bb = rng.multinomial(cb.sum(), cb / cb.sum(), B)
    return decomposition(bl, bb)


def mean_diff(y1, y0, z=1.96):
    """Difference in means with a Welch standard error."""
    d = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return float(d), float(d - z * se), float(d + z * se)
