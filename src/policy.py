"""Who should see the loan, the card or both? Learn a targeting policy and value it honestly.

Arms are coded 0 = loan only (status quo), 1 = card only, 2 = both.

    segment_policy     27 segments (risk x card-suited spend x lump-sum need, thirds of
                       each). For each segment, estimate each arm's value relative to
                       the loan arm and pick the best. Optionally shrink each segment's
                       difference toward the average (empirical-Bayes normal-normal
                       partial pooling), so small noisy segments are not over-trusted.
    regression_policy  per-arm least squares on a quadratic basis with interactions
                       (a "T-learner"); pick the arm with the highest predicted value.
    policy_value       unbiased value of any policy from randomised data: for each arm,
                       the mean over that arm's customers of y * 1{policy chose that arm}.

Scoring a policy on the same data used to choose it is optimistic (winner's curse),
so policies are learned on one half and valued on the other.
"""
import numpy as np
from scipy.stats import beta, norm

N_ARMS = 3


def segments(X):
    """Segment id 0..26 from risk, card-suited share of spend and lump-sum need."""
    risk, card_share, lump = X[:, 0], X[:, 2], X[:, 3]
    r = np.digitize(risk, norm.ppf([1 / 3, 2 / 3]))
    c = np.digitize(card_share, beta.ppf([1 / 3, 2 / 3], 2, 3))
    s = np.digitize(lump, norm.ppf([1 / 3, 2 / 3]))
    return 9 * r + 3 * c + s


def eb_shrink(d, se2):
    """Normal-normal empirical Bayes: shrink estimates d (with variances se2) toward a
    precision-weighted mean. Between-segment variance tau2 by method of moments."""
    tau2 = max(0.0, float(np.var(d, ddof=1) - np.mean(se2)))
    w = 1 / (se2 + tau2)
    mu = float((w * d).sum() / w.sum())
    k = tau2 / (tau2 + se2)
    return mu + k * (d - mu), tau2


def segment_effects(y, arm, seg, n_seg=27, shrink=True):
    """(n_seg, N_ARMS) value of each arm relative to the loan arm, per segment."""
    eff = np.zeros((n_seg, N_ARMS))
    for a in range(1, N_ARMS):
        d, se2 = np.zeros(n_seg), np.zeros(n_seg)
        for s in range(n_seg):
            y1, y0 = y[(seg == s) & (arm == a)], y[(seg == s) & (arm == 0)]
            d[s] = y1.mean() - y0.mean()
            se2[s] = y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)
        eff[:, a] = eb_shrink(d, se2)[0] if shrink else d
    return eff


def segment_policy(y, arm, X, X_new, shrink=True):
    eff = segment_effects(y, arm, segments(X), shrink=shrink)
    return eff.argmax(1)[segments(X_new)]


def basis(X):
    Z = (X - np.array([0.0, 0.0, 0.4, 0.0])) / np.array([1.0, 0.7, 0.2, 1.0])
    cols = [np.ones(len(Z))] + [Z[:, j] for j in range(4)] + [Z[:, j] ** 2 for j in range(4)]
    cols += [Z[:, i] * Z[:, j] for i in range(4) for j in range(i + 1, 4)]
    return np.column_stack(cols)


def regression_predict(y, arm, X, X_new, ridge=1.0):
    """(n_new, N_ARMS) predicted value under each arm."""
    F, F_new = basis(X), basis(X_new)
    pen = ridge * np.eye(F.shape[1])
    pen[0, 0] = 0.0
    out = np.zeros((len(X_new), N_ARMS))
    for a in range(N_ARMS):
        m = arm == a
        coef = np.linalg.solve(F[m].T @ F[m] + pen, F[m].T @ y[m])
        out[:, a] = F_new @ coef
    return out


def regression_policy(y, arm, X, X_new, ridge=1.0):
    return regression_predict(y, arm, X, X_new, ridge).argmax(1)


def policy_value(y, arm, choice):
    """Estimated mean value per customer if everyone got the arm the policy chose."""
    return float(sum((y[arm == a] * (choice[arm == a] == a)).mean() for a in range(N_ARMS)))
