import logging

PAPER_EXACT_OVERRIDES = {
    # Paper Section 2.4
    "model_lr": 0.0005,
    "view_lr": 0.0005,
    "num_gc_layers": 2,
    "emb_dim": 32,
    "mlp_edge_model_dim": 64,
    "batch_size": 32,
    "reg_lambda": 2.0,
    "cr_lambda": 0.4,
    "max_length": 256,

    # Paper equations
    "concrete_temperature": 1.0,
    "batch_temperature": 1.0,
    "memory_temperature": 1.0,
    "contrastive_symmetric": False,
    "regularizer_mode": "paper_keep",

    # Paper-literal GIN
    "normalize_nodes": False,
    "message_relu": False,
    "post_bn_relu": False,
    "drop_ratio": 0.0,

    # Paper representation and evaluation
    "eval_representation": "z",
    "n_folds": 5,
    "feature_type": "instance",
}


def apply_training_profile(args, cli_defaults=None):
    """cli_defaults, when provided (see arg_parse()'s _cli_defaults), maps
    each overridable name to the argparse default it would have had if the
    user hadn't passed it on the command line. Used only to decide whether
    to warn -- every value in the table is still applied unconditionally."""
    if args.training_profile != "paper_exact":
        return args

    cli_defaults = cli_defaults or {}
    for name, value in PAPER_EXACT_OVERRIDES.items():
        current = getattr(args, name, None)
        default = cli_defaults.get(name, current)
        was_explicit = current != default
        would_change = current != value
        if was_explicit and would_change:
            logging.warning(
                "paper_exact profile: --%s=%r was set explicitly on the command line but is "
                "overridden to the paper's value %r under this profile", name, current, value)
        setattr(args, name, value)

    return args
