import math

from agplan.grammar_learning.pcfg import PCFGPrior


def test_uniform_when_unseen():
    p = PCFGPrior(alpha=1.0)
    p.update_rule("S", ("A",))
    p.update_rule("S", ("B",))
    # Unseen rhs falls back to add-alpha smoothing over the LHS's RHS vocab.
    lp = p.logprob("S", ("C",))
    assert lp < 0
    assert math.isfinite(lp)


def test_more_evidence_increases_logprob():
    p = PCFGPrior(alpha=0.1)
    p.update_rule("S", ("A",))
    p.update_rule("S", ("B",))
    lp1 = p.logprob("S", ("A",))
    for _ in range(10):
        p.update_rule("S", ("A",))
    lp2 = p.logprob("S", ("A",))
    assert lp2 > lp1


def test_normalisation_close_to_one():
    p = PCFGPrior(alpha=0.5)
    rhs_list = [("A",), ("B",), ("C",)]
    for rhs in rhs_list:
        p.update_rule("S", rhs, weight=2.0)
    total = sum(math.exp(p.logprob("S", rhs)) for rhs in rhs_list)
    assert abs(total - 1.0) < 1e-9
