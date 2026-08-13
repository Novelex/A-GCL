"""
Tests for unsupervised/training_profiles.py: apply_training_profile() must
leave args untouched for the 'corrected' profile, and overwrite exactly the
documented PAPER_EXACT_OVERRIDES keys (nothing more, nothing less) for
'paper_exact'.
"""
import argparse

import pytest

from unsupervised.training_profiles import PAPER_EXACT_OVERRIDES, apply_training_profile


def make_args(training_profile, **extra):
    args = argparse.Namespace(training_profile=training_profile)
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def test_corrected_profile_leaves_args_completely_unchanged():
    original_values = {name: object() for name in PAPER_EXACT_OVERRIDES}
    args = make_args('corrected', **original_values)

    returned = apply_training_profile(args)

    assert returned is args
    for name, value in original_values.items():
        assert getattr(args, name) is value


def test_paper_exact_profile_overrides_every_documented_key():
    # start every overridable field at a sentinel that's guaranteed to
    # differ from the paper-exact value, so a no-op bug can't hide
    args = make_args('paper_exact', **{name: 'sentinel-not-paper-value' for name in PAPER_EXACT_OVERRIDES})

    apply_training_profile(args)

    for name, expected_value in PAPER_EXACT_OVERRIDES.items():
        assert getattr(args, name) == expected_value


def test_paper_exact_does_not_touch_attributes_outside_the_override_table():
    args = make_args('paper_exact', checkpoint_path='some/user/path.pt', seed=42)
    apply_training_profile(args)
    assert args.checkpoint_path == 'some/user/path.pt'
    assert args.seed == 42


def test_apply_training_profile_is_idempotent():
    args = make_args('paper_exact', **{name: 'sentinel' for name in PAPER_EXACT_OVERRIDES})
    apply_training_profile(args)
    first_pass = vars(args).copy()
    apply_training_profile(args)
    assert vars(args) == first_pass


def test_unknown_profile_value_is_treated_as_not_paper_exact():
    """apply_training_profile only special-cases the literal string
    'paper_exact' -- argparse's choices= already restricts the CLI to valid
    values, but the function itself must not silently misbehave if given
    something else (e.g. during direct unit testing)."""
    args = make_args('not-a-real-profile', reg_lambda=99.0)
    apply_training_profile(args)
    assert args.reg_lambda == 99.0
