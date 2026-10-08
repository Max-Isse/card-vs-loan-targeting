"""Reproduce every number and figure in the README:  python run_all.py"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import beta, norm

from src import decompose, policy, simulate, style

N_WORLDS = 200
GROUPS = ("low", "middle", "high")  # thirds of card-suited share of spend
POLICIES = {
    "all_card": "Card to everyone",
    "all_both": "Both to everyone",
    "seg_raw": "Segments, raw estimates",
    "seg_eb": "Segments, empirical-Bayes pooled",
    "regression": "Regression (T-learner)",
    "seg_oracle": "Best segment rule (truth known)",
    "oracle": "Best arm per customer (truth known)",
}


def card_group(card_share):
    return np.digitize(card_share, beta.ppf([1 / 3, 2 / 3], 2, 3))


def ci(draws, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def true_decomposition(pop, mask=None):
    m = slice(None) if mask is None else mask
    PL = simulate.choice_probs(pop.subset(m) if mask is not None else pop, "loan")
    PB = simulate.choice_probs(pop.subset(m) if mask is not None else pop, "both")
    return {k: float(v) for k, v in decompose.decomposition(PL.mean(0), PB.mean(0)).items()}


def single_world(tpop, tr):
    d = simulate.simulate_experiment(seed=7)
    y, arm, out, pop = d["y"], d["arm"], d["outcome"], d["pop"]
    rows = []
    for a, name in enumerate(simulate.ARMS):
        s = decompose.shares(decompose.counts(out, arm == a))
        ya = y[arm == a]
        se = ya.std(ddof=1) / np.sqrt(len(ya))
        rows.append(
            {
                "arm": name,
                "customers": int((arm == a).sum()),
                **{f"takeup_{k}": float(v) for k, v in s.items()},
                "value": float(ya.mean()),
                "value_ci_low": float(ya.mean() - 1.96 * se),
                "value_ci_high": float(ya.mean() + 1.96 * se),
                "true_value": tr[name]["value"],
            }
        )
    arms = pd.DataFrame(rows)
    arms.to_csv("results/arm_table.csv", index=False)

    diffs = {}
    for a, name in ((1, "card"), (2, "both")):
        dd, lo, hi = decompose.mean_diff(y[arm == a], y[arm == 0])
        diffs[f"{name}_minus_loan"] = {"estimate": dd, "ci_low": lo, "ci_high": hi, "truth": tr[name]["value"] - tr["loan"]["value"]}

    grp, tgrp = card_group(pop.card_share), card_group(tpop.card_share)
    dec_rows = []
    for g, gname in [(None, "all")] + list(enumerate(GROUPS)):
        m = np.ones(len(y), bool) if g is None else grp == g
        cl, cb = decompose.counts(out, m & (arm == 0)), decompose.counts(out, m & (arm == 2))
        est = decompose.decomposition(cl, cb)
        boot = decompose.bootstrap(cl, cb, seed=0 if g is None else g + 1)
        truth = true_decomposition(tpop, None if g is None else tgrp == g)
        for k in ("new", "diverted", "took_both", "card_takeup_both_arm", "loan_takeup_loan_arm", "loan_takeup_both_arm"):
            lo, hi = ci(boot[k])
            dec_rows.append({"group": gname, "quantity": k, "estimate": float(est[k]), "ci_low": lo, "ci_high": hi, "truth": truth[k]})
    dec = pd.DataFrame(dec_rows)
    dec.to_csv("results/decomposition.csv", index=False)
    return d, arms, diffs, dec


def evaluate_policies(d, seed):
    """Learn on a random half, value on the other half. Returns per-policy
    (true gain, held-out estimate, in-sample estimate) vs loan-for-everyone."""
    X, y, arm, pop = d["X"], d["y"], d["arm"], d["pop"]
    mu = np.column_stack([simulate.expected_value(pop, a) for a in simulate.ARMS])
    tr = np.random.default_rng(seed).random(len(y)) < 0.5
    te = ~tr
    seg = policy.segments(X)
    seg_truth = np.array([mu[seg == k].mean(0) if (seg == k).any() else np.zeros(3) for k in range(27)])
    choices = {
        "all_card": np.full(len(y), 1),
        "all_both": np.full(len(y), 2),
        "seg_raw": policy.segment_policy(y[tr], arm[tr], X[tr], X, shrink=False),
        "seg_eb": policy.segment_policy(y[tr], arm[tr], X[tr], X, shrink=True),
        "regression": policy.regression_policy(y[tr], arm[tr], X[tr], X),
        "seg_oracle": seg_truth.argmax(1)[seg],
        "oracle": mu.argmax(1),
    }
    # error of the estimated segment effects of "both" (vs loan), raw and pooled
    seg_true_effect = seg_truth[:, 2] - seg_truth[:, 0]
    est_err = {
        k: float(np.sqrt(((policy.segment_effects(y[tr], arm[tr], seg[tr], shrink=sh)[:, 2] - seg_true_effect) ** 2).mean()))
        for k, sh in (("raw", False), ("eb", True))
    }
    base_te = policy.policy_value(y[te], arm[te], np.zeros(te.sum(), int))
    base_tr = policy.policy_value(y[tr], arm[tr], np.zeros(tr.sum(), int))
    res = {}
    for k, ch in choices.items():
        res[k] = (
            float((mu[te, ch[te]] - mu[te, 0]).mean()),
            policy.policy_value(y[te], arm[te], ch[te]) - base_te,
            policy.policy_value(y[tr], arm[tr], ch[tr]) - base_tr,
        )
    return res, choices, est_err


def monte_carlo(tpop, n_worlds=N_WORLDS):
    tdec = true_decomposition(tpop)
    store = {k: [] for k in POLICIES}
    cover = {"new": [], "diverted": []}
    seg_err = {"raw": [], "eb": []}
    for s in range(1000, 1000 + n_worlds):
        d = simulate.simulate_experiment(seed=s)
        res, _, err = evaluate_policies(d, seed=s)
        for k in seg_err:
            seg_err[k].append(err[k])
        for k, v in res.items():
            store[k].append(v)
        cl, cb = decompose.counts(d["outcome"], d["arm"] == 0), decompose.counts(d["outcome"], d["arm"] == 2)
        boot = decompose.bootstrap(cl, cb, B=1000, seed=s)
        for k in cover:
            lo, hi = ci(boot[k])
            cover[k].append(lo <= tdec[k] <= hi)
    rows = []
    oracle_gain = np.mean([v[0] for v in store["oracle"]])
    for k, v in store.items():
        v = np.array(v)
        rows.append(
            {
                "policy": k,
                "label": POLICIES[k],
                "true_gain": v[:, 0].mean(),
                "true_gain_p10": np.quantile(v[:, 0], 0.1),
                "true_gain_p90": np.quantile(v[:, 0], 0.9),
                "share_of_oracle": v[:, 0].mean() / oracle_gain,
                "share_worlds_beats_status_quo": (v[:, 0] > 0).mean(),
                "heldout_estimate": v[:, 1].mean(),
                "heldout_bias": (v[:, 1] - v[:, 0]).mean(),
                "insample_estimate": v[:, 2].mean(),
                "insample_bias": (v[:, 2] - v[:, 0]).mean(),
            }
        )
    pol = pd.DataFrame(rows)
    pol.to_csv("results/policy_monte_carlo.csv", index=False)
    extra = {f"coverage95_{k}": float(np.mean(v)) for k, v in cover.items()}
    extra.update({f"rmse_segment_both_effect_{k}": float(np.mean(v)) for k, v in seg_err.items()})
    return pol, extra


def figures(d, arms, dec, pol, choices_example):
    style.apply()
    C_NEW, C_DIV, C_BOTH = style.AQUA, style.ORANGE, style.BLUE
    # 1. cannibalisation
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.45, 1]})
    groups = ["all"] + list(GROUPS)
    labels = ["All customers", "Low card-suited spend", "Middle", "High card-suited spend"]
    yy = np.arange(len(groups))
    for i, g in enumerate(groups):
        e = dec[dec["group"] == g].set_index("quantity")["estimate"]
        left = 0.0
        for k, col, lab in (("new", C_NEW, "New customers"), ("diverted", C_DIV, "Loan take-up lost"), ("took_both", C_BOTH, "Took both")):
            v = e[k] * 100
            if v < 0:  # showing two products lost customers overall in this group
                a1.barh(i, v, left=0, color=col, height=0.55, alpha=0.6)
                a1.text(v - 0.3, i, f"{v:.1f}", ha="right", va="center", fontsize=8, color=style.INK2)
                continue
            a1.barh(i, v, left=left, color=col, height=0.55, label=lab if i == 0 else None)
            if v >= 1.2:
                a1.text(left + v / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white")
            left += v
        a1.text(left + 0.4, i, f"{e['card_takeup_both_arm']:.1%} take the card", va="center", fontsize=8.5, color=style.INK2)
    a1.axvline(0, color=style.AXIS, linewidth=1)
    a1.set_yticks(yy)
    a1.set_yticklabels(labels)
    a1.invert_yaxis()
    a1.set_xlim(-4, 36)
    a1.set_xlabel("% of customers shown both")
    a1.set_title("Where card takers come from (both vs loan only)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    a1.grid(axis="y", visible=False)

    x = np.arange(len(arms))
    a2.bar(x, arms["value"], color=[style.MUTED, style.ORANGE, style.BLUE], width=0.55)
    a2.vlines(x, arms["value_ci_low"], arms["value_ci_high"], color=style.INK2, linewidth=1.2)
    a2.plot(x, arms["true_value"], "D", color=style.INK, markersize=5, label="Truth")
    for i, r in arms.iterrows():
        a2.text(i, r["value"] * 0.5, f"£{r['value']:.0f}", ha="center", va="center", fontsize=9, color="white")
    a2.set_xticks(x)
    a2.set_xticklabels(["Loan only\n(status quo)", "Card only", "Both"])
    a2.set_ylabel("12-month contribution per customer (£)")
    a2.set_title("Value per customer shown (95% CI)")
    a2.legend(loc="upper center", fontsize=8.5)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("figures/cannibalisation.png")
    plt.close(fig)

    # 2. policy value
    order = ["all_card", "all_both", "seg_eb", "seg_raw", "regression", "seg_oracle", "oracle"]
    p = pol.set_index("policy").loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    yy = np.arange(len(order))
    cols = [style.MUTED if k in ("all_card", "all_both") else style.GRID if k in ("seg_oracle", "oracle") else style.BLUE for k in order]
    ax.barh(yy, p["true_gain"], color=cols, height=0.55, label="True gain (mean of 200 tests)")
    ax.hlines(yy, p["true_gain_p10"], p["true_gain_p90"], color=style.INK2, linewidth=1.2)
    learned = [k in ("seg_raw", "seg_eb", "regression") for k in order]
    ax.plot(p["heldout_estimate"][learned], yy[learned] - 0.0, "o", color=style.AQUA, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.2, label="Estimate on held-out half")
    ax.plot(p["insample_estimate"][learned], yy[learned], "X", color=style.ORANGE, markersize=8, markeredgecolor=style.SURFACE, markeredgewidth=0.8, label="Estimate on the data used to learn it")
    ax.axvline(0, color=style.AXIS, linewidth=1)
    ax.set_xlim(-60, 145)  # card-to-everyone is far off the left edge; label it instead
    ax.text(-57, 0, f"← −£{-p.loc['all_card', 'true_gain']:,.0f}", va="center", fontsize=8.5, color="white")
    for i, k in enumerate(order[1:], start=1):
        ax.text(p.loc[k, "true_gain_p90"] + 3, i - 0.33, f"£{p.loc[k, 'true_gain']:.0f}", va="center", fontsize=8.5, color=style.INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([POLICIES[k] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gain per customer vs loan to everyone (£, 12-month contribution)")
    ax.set_title("Targeting policies: true gain vs what the data says")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig("figures/policy_value.png")
    plt.close(fig)

    # 3. policy map
    pop = d["pop"]
    band = np.digitize(pop.risk, norm.ppf([1 / 3, 2 / 3]))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(pop.risk), 6000, replace=False)
    cmap = ListedColormap([style.MUTED, style.ORANGE, style.BLUE])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    for r, (key, ttl) in enumerate((("oracle", "Best arm (truth known)"), ("regression", "Regression policy (learned)"))):
        ch = choices_example[key]
        for b, bname in enumerate(("Low", "Medium", "High")):
            ax = axes[r, b]
            m = pick[band[pick] == b]
            ax.scatter(pop.card_share[m], pop.lump[m], c=ch[m], cmap=cmap, vmin=0, vmax=2, s=5, linewidths=0, alpha=0.8)
            ax.set_title(f"{ttl}\n{bname} risk", fontsize=9.5)
            ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("Share of spend suited to a card")
    for ax in axes[:, 0]:
        ax.set_ylabel("Lump-sum need (std.)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=6) for c in (style.MUTED, style.ORANGE, style.BLUE)]
    fig.legend(handles, ["Loan only", "Card only", "Both"], loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("figures/policy_map.png")
    plt.close(fig)


def main():
    tpop, tr = simulate.truth()
    print("\nTRUTH\n", json.dumps({k: {"probs": v["probs"].round(4).tolist(), "value": v["value"]} for k, v in tr.items()}, indent=1))
    d, arms, diffs, dec = single_world(tpop, tr)
    print("\nARMS\n", arms.round(4).to_string())
    print("\nVALUE DIFFERENCES\n", json.dumps(diffs, indent=1))
    print("\nDECOMPOSITION (both vs loan only)\n", dec.round(4).to_string())
    pol, cover = monte_carlo(tpop)
    print("\nPOLICIES (200 simulated tests)\n", pol.round(3).to_string())
    print("\nDECOMPOSITION CI COVERAGE AND SEGMENT-EFFECT ERROR\n", cover)
    # example-world policy map: learn on the full experiment
    X, y, arm = d["X"], d["y"], d["arm"]
    mu = np.column_stack([simulate.expected_value(d["pop"], a) for a in simulate.ARMS])
    choices = {"oracle": mu.argmax(1), "regression": policy.regression_policy(y, arm, X, X)}
    agree = float((choices["oracle"] == choices["regression"]).mean())
    shares = {k: (np.bincount(v, minlength=3) / len(v)).tolist() for k, v in choices.items()}
    print("\nEXAMPLE WORLD: arm shares", shares, "agreement", round(agree, 3))
    band = np.digitize(d["pop"].risk, norm.ppf([1 / 3, 2 / 3]))
    card_by_band = {k: [float((v[band == b] == 1).mean()) for b in range(3)] for k, v in choices.items()}
    print("card-only share by risk band (low, medium, high)", card_by_band)
    v_loan, v_card = simulate.expected_product_values(tpop)
    medians = {"loan": float(np.median(v_loan)), "card": float(np.median(v_card))}
    print("median expected 12-month value of a product taken", medians)
    figures(d, arms, dec, pol, choices)
    with open("results/results.json", "w") as f:
        json.dump(
            {
                "truth": {k: {"probs": v["probs"].tolist(), "value": v["value"]} for k, v in tr.items()},
                "arms": arms.to_dict("records"),
                "value_differences": diffs,
                "decomposition": dec.to_dict("records"),
                "policies": pol.to_dict("records"),
                "decomposition_ci_coverage": cover,
                "example_policy_shares": shares,
                "example_policy_agreement_with_oracle": agree,
                "example_card_only_share_by_risk_band": card_by_band,
                "median_product_value": medians,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
