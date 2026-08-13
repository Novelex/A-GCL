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


def apply_training_profile(args):
    if args.training_profile != "paper_exact":
        return args

    for name, value in PAPER_EXACT_OVERRIDES.items():
        setattr(args, name, value)

    return args
