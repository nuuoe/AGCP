import math

from agplan.grammar_learning.adaptor_cache import AdaptorCache


def test_failed_plans_do_not_update():
    c = AdaptorCache()
    c.update(("a", "b"), success=False)
    assert c.total == 0
    assert c.fragments == {}


def test_repeated_success_increases_prior():
    c = AdaptorCache(discount=0.5, strength=1.0)
    frag = ("turn_left", "move_forward")
    c.update(frag, success=True)
    lp1 = c.log_prior(frag)
    for _ in range(10):
        c.update(frag, success=True)
    lp2 = c.log_prior(frag)
    assert lp2 > lp1
    assert math.isfinite(lp1)
    assert math.isfinite(lp2)


def test_unseen_fragment_returns_finite_prior():
    c = AdaptorCache()
    c.update(("a", "b"), success=True)
    lp = c.log_prior(("x", "y"))
    assert math.isfinite(lp)
    assert lp < 0
