"""
Tests for the regularizer sign convention used in the training loop's view
update: view_loss adds reg_sign * reg_lambda * reg, and the view learner is
then updated via gradient ASCENT on view_loss ((-view_loss).backward()
followed by optimizer.step()). For 'paper_keep' (reg = mean(mu), the paper's
own R(mu) definition, reg_sign=-1), ascending view_loss must push mu DOWN --
more dropping -- since a larger reg makes view_loss smaller: no term opposes
the (simulated, absent here) contrastive loss's drop-pressure, so this
isolates the collapse property documented in docs/changes.md. For 'budget'
(same reg, reg_sign=+1), ascending view_loss must push mu UP instead -- the
AD-GCL perturbation-budget reading, which in the real training loop (with
the real contrastive term present) opposes that drop-pressure rather than
reinforcing it. These tests isolate just the sign convention and cannot by
themselves prove convergence to an interior equilibrium under the full
adversarial objective -- that requires the real (nonlinear) contrastive
term, which only a full training run exercises; see docs/changes.md and
correction.md Section 10 for the empirical keep-probability trajectories.
"""
import torch
import pytest


def ascend_view_loss(logits, reg_lambda, contrastive_term=None, reg_sign=-1.0):
    """Mirrors the training loop's view-update half-step exactly enough to
    test the regularizer's sign in isolation: view_loss = contrastive + reg_sign*reg_lambda*reg,
    then (-view_loss).backward() + an SGD step, as agcl_ABIDE(_queue).py does.
    Returns mu AFTER the step -- callers comparing before/after must not use
    a mu snapshot taken before optimizer.step() actually moved logits."""
    optimizer = torch.optim.SGD([logits], lr=1.0)
    optimizer.zero_grad()

    mu = torch.sigmoid(logits)
    reg = mu.mean()  # R(mu) = mean(mu), shared by paper_keep and budget
    view_loss = (contrastive_term if contrastive_term is not None else torch.zeros(())) + reg_sign * reg_lambda * reg
    (-view_loss).backward()
    optimizer.step()
    return torch.sigmoid(logits).detach()


def test_paper_regularizer_reduces_keep_probability():
    torch.manual_seed(0)
    logits = torch.zeros(100, requires_grad=True)
    mu_before = torch.sigmoid(logits).mean().item()

    ascend_view_loss(logits, reg_lambda=2.0)

    mu_after = torch.sigmoid(logits).mean().item()
    assert mu_after < mu_before


def test_larger_reg_lambda_drops_keep_probability_faster():
    torch.manual_seed(0)
    logits_small_lambda = torch.zeros(100, requires_grad=True)
    logits_large_lambda = torch.zeros(100, requires_grad=True)

    mu_small = ascend_view_loss(logits_small_lambda, reg_lambda=0.5).mean().item()
    mu_large = ascend_view_loss(logits_large_lambda, reg_lambda=5.0).mean().item()

    # both started at mu=0.5; the larger lambda must have dropped further
    assert mu_large < mu_small


def test_regularizer_term_alone_has_no_stationary_point_at_positive_keep_prob():
    """This is the collapse behavior documented in docs/changes.md: with no
    opposing pressure, repeatedly ascending view_loss with only the
    regularizer term present drives mu monotonically toward 0, never
    settling -- unlike a 'budget'-style regularizer that would stabilize."""
    torch.manual_seed(0)
    logits = torch.zeros(20, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.5)

    keep_probs = []
    for _ in range(10):
        optimizer.zero_grad()
        mu = torch.sigmoid(logits)
        reg = mu.mean()
        view_loss = -2.0 * reg
        (-view_loss).backward()
        optimizer.step()
        keep_probs.append(mu.mean().item())

    # strictly decreasing at every step -- no equilibrium reached
    assert all(earlier > later for earlier, later in zip(keep_probs, keep_probs[1:]))


def test_target_keep_ablation_has_a_stationary_point_at_the_target():
    """Contrast case: the --regularizer_mode target_keep ablation (reg =
    (keep_prob - target)^2) DOES have a stationary point once mu reaches the
    target, unlike paper_keep above -- confirming the two regularizer_mode
    options really do behave differently, not just differ in name."""
    torch.manual_seed(0)
    target = 0.2
    logits = torch.zeros(200, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.5)

    # reg's gradient w.r.t. each logit is diluted by 1/200 (mean over 200
    # elements) -- 500 iterations was not enough to close the gap (landed
    # near 0.42, not 0.2); 5000 gives ample margin
    for _ in range(5000):
        optimizer.zero_grad()
        mu = torch.sigmoid(logits)
        keep_prob = mu.mean()
        reg = (keep_prob - target).pow(2)
        view_loss = -2.0 * reg
        (-view_loss).backward()
        optimizer.step()

    final_keep_prob = torch.sigmoid(logits).mean().item()
    assert abs(final_keep_prob - target) < 0.05


def test_budget_regularizer_increases_keep_probability():
    """budget mode (reg_sign=+1): the opposite direction from paper_keep --
    ascending pushes mu UP, since a larger keep-probability now makes
    view_loss larger instead of smaller."""
    torch.manual_seed(0)
    logits = torch.zeros(100, requires_grad=True)
    mu_before = torch.sigmoid(logits).mean().item()

    ascend_view_loss(logits, reg_lambda=2.0, reg_sign=1.0)

    mu_after = torch.sigmoid(logits).mean().item()
    assert mu_after > mu_before


def test_budget_and_paper_keep_push_in_opposite_directions():
    """Same starting point, same reg_lambda, only reg_sign differs -- the
    two must move symmetrically in opposite directions. Confirms 'budget'
    and 'paper_keep' aren't just different names for the same update."""
    torch.manual_seed(0)
    logits_budget = torch.zeros(50, requires_grad=True)
    logits_paper_keep = torch.zeros(50, requires_grad=True)

    mu_budget = ascend_view_loss(logits_budget, reg_lambda=1.0, reg_sign=1.0).mean().item()
    mu_paper_keep = ascend_view_loss(logits_paper_keep, reg_lambda=1.0, reg_sign=-1.0).mean().item()

    assert mu_budget > 0.5 > mu_paper_keep
    # symmetric magnitude: same starting point (mu=0.5), same |reg_lambda|,
    # opposite sign -> equal and opposite displacement
    assert abs((mu_budget - 0.5) - (0.5 - mu_paper_keep)) < 1e-6


def test_regularizer_term_alone_has_no_stationary_point_at_positive_keep_prob_budget_mode():
    """Mirrors test_regularizer_term_alone_has_no_stationary_point_at_positive_keep_prob
    but for budget mode: with no opposing (contrastive) pressure, the
    regularizer term ALONE still has no interior stationary point either --
    it now drives mu monotonically toward 1 instead of 0. The interior
    equilibrium budget mode is meant to produce only exists once the real
    (opposing) contrastive term is also present, which these isolated tests
    deliberately don't simulate -- see the module docstring."""
    torch.manual_seed(0)
    logits = torch.zeros(20, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.5)

    keep_probs = []
    for _ in range(10):
        optimizer.zero_grad()
        mu = torch.sigmoid(logits)
        reg = mu.mean()
        view_loss = 2.0 * reg
        (-view_loss).backward()
        optimizer.step()
        keep_probs.append(mu.mean().item())

    # strictly increasing at every step -- no equilibrium reached without an
    # opposing force
    assert all(earlier < later for earlier, later in zip(keep_probs, keep_probs[1:]))
