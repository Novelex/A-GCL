"""
Tests for the regularizer sign convention used in the training loop's view
update: view_loss subtracts reg_lambda * reg, and the view learner is then
updated via gradient ASCENT on view_loss ((-view_loss).backward() followed
by optimizer.step()). For 'paper_keep' (reg = mean(mu), the paper's own
R(mu) definition), ascending view_loss must push mu DOWN -- more dropping --
since a larger reg makes view_loss smaller. This isolates just the sign
convention, independent of the rest of the training loop.
"""
import torch
import pytest


def ascend_view_loss(logits, reg_lambda, contrastive_term=None):
    """Mirrors the training loop's view-update half-step exactly enough to
    test the regularizer's sign in isolation: view_loss = contrastive - reg_lambda*reg,
    then (-view_loss).backward() + an SGD step, as agcl_ABIDE(_queue).py does."""
    optimizer = torch.optim.SGD([logits], lr=1.0)
    optimizer.zero_grad()

    mu = torch.sigmoid(logits)
    reg = mu.mean()  # paper_keep: R(mu) = mean(mu)
    view_loss = (contrastive_term if contrastive_term is not None else torch.zeros(())) - reg_lambda * reg
    (-view_loss).backward()
    optimizer.step()
    return mu.detach()


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

    for _ in range(500):
        optimizer.zero_grad()
        mu = torch.sigmoid(logits)
        keep_prob = mu.mean()
        reg = (keep_prob - target).pow(2)
        view_loss = -2.0 * reg
        (-view_loss).backward()
        optimizer.step()

    final_keep_prob = torch.sigmoid(logits).mean().item()
    assert abs(final_keep_prob - target) < 0.05
