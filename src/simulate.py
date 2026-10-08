"""Simulate a three-arm offer test: flexible loan only, credit card only, or both.

Real lending data is private, so this repo uses a synthetic population whose true
behaviour is known. That lets every estimate and every targeting policy be scored
against ground truth.

Each eligible business has two latent needs:
    loan utility  u_L: driven by an upcoming lump-sum need (stock, equipment, a tax bill)
    card utility  u_C: driven by the share of outgoings that suit a card (software, ads, travel)
Shown one product, it takes it with probability expit(u). Shown both, it chooses among
{neither, loan, card, both} by multinomial logit, where taking both carries a penalty
KAPPA. With KAPPA = 0 (and no ATTENTION cost) the two decisions are independent and the
card cannot take customers from the loan; the larger KAPPA, the more the products
substitute. ATTENTION lowers each product's utility when two are shown at once
(divided attention, a harder choice).

Value is a 12-month contribution (margin minus expected credit loss). Loans are
larger and earn more per customer; cards earn interchange and interest on a smaller
limit. All parameters are my assumptions, not estimates.
"""
from dataclasses import dataclass

import numpy as np

ARMS = ("loan", "card", "both")
KAPPA = 1.5
ATTENTION = 0.3
LOAN_MARGIN = 0.12  # 12-month net interest margin, share of loan amount
LOAN_LOSS = 0.48  # loss given default x average exposure, share of amount
CARD_INTERCHANGE = 0.01
CARD_INTEREST = 0.10  # net interest over 12 months, share of limit
CARD_COST = 60.0  # servicing cost per card, £
CARD_LOSS = 0.54  # loss given default x average utilisation, share of limit
OUTCOMES = ("none", "loan", "card", "both")


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Population:
    risk: np.ndarray  # standardised credit risk, higher = riskier
    turnover: np.ndarray  # monthly, £
    card_share: np.ndarray  # share of outgoings suited to a card
    lump: np.ndarray  # standardised upcoming lump-sum need
    u_loan: np.ndarray
    u_card: np.ndarray
    pd12: np.ndarray
    loan_amount: np.ndarray
    card_limit: np.ndarray

    def __len__(self):
        return len(self.risk)

    def subset(self, idx):
        return Population(**{k: v[idx] for k, v in self.__dict__.items()})


def population(n, rng):
    risk = rng.normal(0, 1, n)
    turnover = np.exp(rng.normal(np.log(30_000), 0.7, n))
    card_share = rng.beta(2, 3, n)
    lump = rng.normal(0, 1, n)
    return Population(
        risk=risk,
        turnover=turnover,
        card_share=card_share,
        lump=lump,
        u_loan=-1.4 + 1.0 * lump + 0.2 * risk,
        u_card=-1.8 + 4.0 * (card_share - 0.4) + 0.3 * np.log(turnover / 30_000),
        pd12=expit(-3.0 + 1.0 * risk),
        loan_amount=0.8 * turnover * np.exp(0.15 * lump),
        card_limit=0.25 * turnover,
    )


def features(pop):
    """Pre-treatment covariates a lender could see (for targeting)."""
    return np.column_stack([pop.risk, np.log(pop.turnover / 30_000), pop.card_share, pop.lump])


def choice_probs(pop, arm, kappa=KAPPA, attention=ATTENTION):
    """(n, 4) probabilities of taking none / loan / card / both under an arm."""
    n = len(pop)
    P = np.zeros((n, 4))
    if arm == "loan":
        p = expit(pop.u_loan)
        P[:, 0], P[:, 1] = 1 - p, p
    elif arm == "card":
        p = expit(pop.u_card)
        P[:, 0], P[:, 2] = 1 - p, p
    elif arm == "both":
        ul, uc = pop.u_loan - attention, pop.u_card - attention
        e = np.column_stack([np.zeros(n), ul, uc, ul + uc - kappa])
        e = np.exp(e - e.max(1, keepdims=True))
        P = e / e.sum(1, keepdims=True)
    else:
        raise ValueError(arm)
    return P


def expected_product_values(pop):
    """Expected 12-month contribution of a loan and of a card for each customer."""
    v_loan = pop.loan_amount * (LOAN_MARGIN - pop.pd12 * LOAN_LOSS)
    card_spend = 12 * pop.turnover * pop.card_share * 0.3
    v_card = CARD_INTERCHANGE * card_spend + CARD_INTEREST * pop.card_limit - CARD_COST - pop.pd12 * CARD_LOSS * pop.card_limit
    return v_loan, v_card


def expected_value(pop, arm, kappa=KAPPA, attention=ATTENTION):
    """True expected 12-month contribution per customer if shown `arm`."""
    P = choice_probs(pop, arm, kappa, attention)
    v_loan, v_card = expected_product_values(pop)
    return P[:, 1] * v_loan + P[:, 2] * v_card + P[:, 3] * (v_loan + v_card)


def simulate_experiment(seed=7, n=36_000, kappa=KAPPA, attention=ATTENTION):
    """Complete randomisation, n/3 per arm. Realised value includes default and usage noise."""
    rng = np.random.default_rng(seed)
    pop = population(n, rng)
    arm = np.repeat(np.arange(3), n // 3)
    rng.shuffle(arm)
    P = np.zeros((n, 4))
    for a, name in enumerate(ARMS):
        m = arm == a
        P[m] = choice_probs(pop.subset(m), name, kappa, attention)
    u = rng.random(n)[:, None]
    outcome = (u > P.cumsum(1)).sum(1)  # 0 none, 1 loan, 2 card, 3 both
    has_loan = np.isin(outcome, (1, 3))
    has_card = np.isin(outcome, (2, 3))
    usage = np.exp(rng.normal(-0.045, 0.3, n))  # card usage varies, mean 1
    loan_default = rng.random(n) < pop.pd12
    card_default = rng.random(n) < pop.pd12
    card_spend = 12 * pop.turnover * pop.card_share * 0.3 * usage
    y_loan = pop.loan_amount * LOAN_MARGIN - loan_default * pop.loan_amount * LOAN_LOSS
    y_card = (
        CARD_INTERCHANGE * card_spend
        + CARD_INTEREST * pop.card_limit * usage
        - CARD_COST
        - card_default * CARD_LOSS * pop.card_limit
    )
    y = has_loan * y_loan + has_card * y_card
    return {"pop": pop, "X": features(pop), "arm": arm, "outcome": outcome, "y": y}


def truth(n=1_000_000, seed=99):
    """Population-level expected take-up and value under each arm, from a large sample."""
    pop = population(n, np.random.default_rng(seed))
    out = {}
    for a in ARMS:
        out[a] = {"probs": choice_probs(pop, a).mean(0), "value": float(expected_value(pop, a).mean())}
    return pop, out
