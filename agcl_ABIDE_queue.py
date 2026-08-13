import argparse
import logging
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC, SVC
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from datasets import ABIDEDataset
from datasets import TUEvaluator
from unsupervised.embedding_evaluation import (EmbeddingEvaluation, get_emb_y, create_fixed_splits,
                                                 paper_five_fold_evaluation)
from unsupervised.encoder import TUEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.view_learner import (ViewLearner, symmetrize_edge_logits, compute_reverse_index,
                                       sample_symmetric_logistic_noise, sample_ordered_concrete_mask)
from unsupervised.utils import initialize_node_features, set_tu_dataset_y_shape
from unsupervised.training_profiles import apply_training_profile
from numpy import interp


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected true/false, got %r' % value)


def calc_regloss(z, aug, memory, memory_subject_ids, anchor_subject_ids, temperature: float = 0.1, pos_only: bool = False):

	z = F.normalize(z, dim=-1)
	aug = F.normalize(aug, dim=-1)
	memory = F.normalize(memory, dim=-1)

	# diagonal (i,i) is the positive pair: anchor i vs. its own augmented view
	logits = torch.einsum("if, jf -> ij", z, aug) / temperature

	m_logits = torch.einsum("if, jf -> ij", z, memory) / temperature
	# a subject's own (earlier-epoch) embedding must not count as its own negative
	same_subject = anchor_subject_ids.view(-1, 1) == memory_subject_ids.view(1, -1)
	has_negative = (~same_subject).any(dim=1)
	if not bool(has_negative.any()):
		return z.sum() * 0.0

	# filter to anchors with at least one valid negative BEFORE logsumexp,
	# so an all-same-subject row never produces an all -inf denominator (and
	# the +inf/NaN that follows from it) in the first place -- rather than
	# computing it and discarding the result afterwards
	positive_logits = logits.diagonal()[has_negative]

	if pos_only:
		mean_log_prob_pos = positive_logits
	else:
		valid_memory_logits = m_logits[has_negative]
		valid_same_subject = same_subject[has_negative]
		valid_memory_logits = valid_memory_logits.masked_fill(valid_same_subject, float('-inf'))
		denominator = torch.logsumexp(valid_memory_logits, dim=1)
		mean_log_prob_pos = positive_logits - denominator

	loss = -mean_log_prob_pos.mean()

	return loss


def calc_regloss_paper(z, aug, memory, temperature: float = 0.1):
	"""paper_exact profile: literal memory-bank contrastive loss. Every row in
	`memory` counts as a negative -- including zero-filled, never-yet-written
	rows before the queue first fills, and a subject's own earlier-epoch
	embedding, since the paper's formulas have no same-subject exclusion.
	Kept alongside, not in place of, the corrected calc_regloss (which adds
	the valid-only queue and same-subject exclusion). The numerically-stable
	logsumexp from bug 7 is still used here -- that fix is pure numerical
	hygiene, not a paper-fidelity tradeoff, so it applies to both profiles."""
	z = F.normalize(z, dim=-1)
	aug = F.normalize(aug, dim=-1)
	memory = F.normalize(memory, dim=-1)

	logits = torch.einsum("if, jf -> ij", z, aug) / temperature
	m_logits = torch.einsum("if, jf -> ij", z, memory) / temperature

	positive_logits = logits.diagonal()
	denominator = torch.logsumexp(m_logits, dim=1)
	mean_log_prob_pos = positive_logits - denominator

	loss = -mean_log_prob_pos.mean()

	return loss


# strategy 1. memory bank without update criteria(only kept the latest features)
class MemoryBank_Q:
    def __init__(self, max_length, feature_dim, device):
        self.max_length = max_length
        self.memory = torch.zeros((max_length, feature_dim)).to(device)
        self.subject_ids = torch.full((max_length,), -1, dtype=torch.long, device=device)
        self.valid = torch.zeros(max_length, dtype=torch.bool, device=device)
        self.current_index = 0

    def push(self, features, subject_ids):
        features = features.detach()
        subject_ids = subject_ids.detach()
        batch_size = features.size(0)
        if (self.current_index + batch_size) < self.max_length:
            written = slice(self.current_index, self.current_index + batch_size)
            self.memory.data[written, :] = features
            self.subject_ids[written] = subject_ids
            self.valid[written] = True
            self.current_index = (self.current_index + batch_size) % self.max_length
        else:
            current_index = batch_size - (self.max_length - self.current_index)
            tail = slice(self.current_index, self.max_length)
            head = slice(0, current_index)
            self.memory.data[tail, :] = features[:self.max_length - self.current_index, :]
            self.subject_ids[tail] = subject_ids[:self.max_length - self.current_index]
            self.valid[tail] = True
            self.memory.data[head, :] = features[self.max_length - self.current_index:, :]
            self.subject_ids[head] = subject_ids[self.max_length - self.current_index:]
            self.valid[head] = True
            self.current_index = current_index

    def get_valid_memory(self):
        return self.memory[self.valid], self.subject_ids[self.valid]


class PaperMemoryBank_Q:
    """paper_exact profile: literal FIFO queue, zero-initialized, with no
    validity mask -- the zero rows count as negatives until the queue first
    fills (see calc_regloss_paper), and no subject_id tracking, since the
    paper's formulas never exclude a subject's own past embedding. Kept
    alongside, not in place of, the corrected MemoryBank_Q."""
    def __init__(self, max_length, feature_dim, device):
        self.max_length = max_length
        self.memory = torch.zeros((max_length, feature_dim)).to(device)
        self.current_index = 0

    def push(self, features):
        features = features.detach()
        batch_size = features.size(0)
        if (self.current_index + batch_size) < self.max_length:
            written = slice(self.current_index, self.current_index + batch_size)
            self.memory.data[written, :] = features
            self.current_index = (self.current_index + batch_size) % self.max_length
        else:
            current_index = batch_size - (self.max_length - self.current_index)
            tail = slice(self.current_index, self.max_length)
            head = slice(0, current_index)
            self.memory.data[tail, :] = features[:self.max_length - self.current_index, :]
            self.memory.data[head, :] = features[self.max_length - self.current_index:, :]
            self.current_index = current_index

    def get_memory(self):
        return self.memory


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run(args):
    args = apply_training_profile(args, cli_defaults=getattr(args, '_cli_defaults', None))
    is_paper_exact = args.training_profile == "paper_exact"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info("Training profile: %s", args.training_profile)
    logging.info(args)
    setup_seed(args.seed)


    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ABIDEDataset(args.path, args.name, transform=my_transforms)

    dataset.data.y = dataset.data.y.squeeze()

    n_subjects = len(dataset)
    n_asd = int((dataset.data.y == 1).sum())
    n_nc = int((dataset.data.y == 0).sum())
    n_rois = dataset[0].x.size(0)
    logging.info("N = %d (ASD = %d, NC = %d)", n_subjects, n_asd, n_nc)
    logging.info("ROIs (M) = %d, edges per graph = %d", n_rois, n_rois * n_rois)

    if is_paper_exact:
        paper_n_subjects, paper_n_rois = 987, 116
        dataset_matches_paper = (n_subjects == paper_n_subjects and n_rois == paper_n_rois)
        if dataset_matches_paper:
            logging.info("Dataset preflight: matches paper's ABIDE-I configuration (987 subjects, 116 ROIs).")
        else:
            msg = (
                "Dataset preflight: paper_exact profile selected, but this dataset is "
                "%d subjects / %d ROIs, not the paper's 987 subjects / 116 ROIs (AAL1). "
                "This run tests the paper-literal CODE PATH, but its result is NOT an "
                "exact reproduction of the paper's ABIDE-I experiment."
            ) % (n_subjects, n_rois)
            if args.allow_dataset_mismatch:
                logging.warning(msg)
            else:
                raise ValueError(
                    msg + " Pass --allow_dataset_mismatch true to run anyway."
                )

    # precomputed once, reused for every evaluation across every epoch --
    # never resampled, so results are reproducible from --seed
    fixed_splits = create_fixed_splits(dataset, n_splits=args.n_folds, seed=args.seed)
    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)

    memory_bank = MemoryBank_Q(max_length=args.max_length, feature_dim=args.emb_dim, device=device)
    paper_memory_bank = (
        PaperMemoryBank_Q(max_length=args.max_length, feature_dim=args.emb_dim, device=device)
        if is_paper_exact else None
    )

    evaluator = TUEvaluator()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=3, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
                  drop_ratio=args.drop_ratio, pooling_type=args.pooling_type,
                  normalize_nodes=args.normalize_nodes, message_relu=args.message_relu, post_bn_relu=args.post_bn_relu),
        args.emb_dim).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    view_learner = ViewLearner(TUEncoder(num_dataset_features=3, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
                                         drop_ratio=args.drop_ratio, pooling_type=args.pooling_type,
                                         normalize_nodes=args.normalize_nodes, message_relu=args.message_relu, post_bn_relu=args.post_bn_relu),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True, max_iter=10000), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True)

    model.eval()
    train_score, val_score, _ = ee.kf_embedding_evaluation(
        model, dataset, fixed_splits, representation=args.eval_representation, include_test=False)
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {}".format(train_score, val_score))

    model_losses = []
    view_losses = []
    keep_probs = []

    # guarantee a checkpoint always exists, even if no in-loop eval ever
    # triggers (e.g. epochs < eval_interval) -- worst case, this is what
    # gets reloaded for the final evaluation
    best_val_accuracy = val_score[0]
    best_epoch = 0
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'view_state_dict': view_learner.state_dict(),
        'model_optimizer': model_optimizer.state_dict(),
        'view_optimizer': view_optimizer.state_dict(),
        'validation_score': val_score,
    }, args.checkpoint_path)


    for epoch in range(1, args.epochs + 1):
        model_loss_all = 0
        view_loss_all = 0
        keep_prob_all = 0
        sampled_keep_all = 0
        for batch in dataloader:
            # set up
            batch = batch.to(device)

            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            model.eval()
        
            x, _ = model(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)

            if is_paper_exact:
                # paper_exact: literal ordered/asymmetric Concrete mask, no
                # reverse-edge symmetrization -- (i,j) and (j,i) may sample
                # to different keep/drop outcomes, matching the printed paper
                mu, edge_mask = sample_ordered_concrete_mask(
                    edge_logits, temperature=args.concrete_temperature)
            else:
                rev_idx = compute_reverse_index(batch.edge_index)
                sym_logits = symmetrize_edge_logits(batch.edge_index, edge_logits, rev_idx)
                mu = torch.sigmoid(sym_logits)

                temperature = args.concrete_temperature
                bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                noise = sample_symmetric_logistic_noise(batch.edge_index, rev_idx, bias=bias, device=device)
                gate_inputs = (noise + sym_logits) / temperature
                edge_mask = torch.sigmoid(gate_inputs)
            aug_edge_weight = batch.edge_weight * edge_mask

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, aug_edge_weight)
            # keep_prob: paper's R(mu) = mean(mu), the deterministic keep-probability
            row, col = batch.edge_index
            edge_batch = batch.batch[row]

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(mu, edge_batch, reduce="sum")
            keep_prob = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    keep_prob.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            keep_prob = torch.stack(keep_prob).mean()

            # three regularizer_mode options, differing in what reg equals AND in
            # the sign it enters view_loss with (reg_sign) -- both matter:
            # 'paper_keep' (paper_exact's forced value) -- reg = R(mu) = mean(mu),
            #   sign '-', exactly as the paper's printed formula literally reads.
            #   No term opposes calc_loss's pressure to drop everything -- keep_prob
            #   collapses monotonically. This is an accepted, documented consequence
            #   of the paper's own formula (see docs/changes.md), not a bug.
            # 'budget' (corrected's default) -- reg = mean(mu) too, but sign '+'.
            #   Ascending +reg_lambda*mean(mu) pushes keep_prob UP, opposing
            #   calc_loss's drop-pressure -- the AD-GCL "perturbation budget"
            #   reading. Mathematically equivalent to the pre-R(mu)-rewrite
            #   convention (-reg_lambda*mean(1-mu), see correction.md Section 2's
            #   update note) reached via a '+' sign instead of redefining reg.
            # 'target_keep' -- engineering ablation, NOT the paper's objective:
            #   pulls keep_prob toward a fixed --target_keep ratio via squared
            #   error; sign '-' converges either way since (keep_prob-target)^2
            #   is minimized regardless of which side keep_prob approaches from.
            if args.regularizer_mode == 'target_keep':
                reg = (keep_prob - args.target_keep).pow(2)
                reg_sign = -1.0
            elif args.regularizer_mode == 'budget':
                reg = keep_prob
                reg_sign = 1.0
            else:
                reg = keep_prob
                reg_sign = -1.0

            sampled_keep = edge_mask.mean()

            if is_paper_exact:
                # paper-literal bug 5: the queue is pushed here, before this
                # step's cr_loss is computed, so this batch's own just-pushed
                # embeddings are already inside the pool they're scored
                # against -- and remain so for the model-update half-step
                # below, since it's the same shared queue object.
                paper_memory_bank.push(x_aug.detach())
                memory = paper_memory_bank.get_memory()
                cr_loss = calc_regloss_paper(x, x_aug, memory, temperature=args.memory_temperature)
            else:
                # memory bank read here uses the state from BEFORE this batch --
                # enqueueing happens once, at the very end, after both updates
                valid_memory, valid_memory_ids = memory_bank.get_valid_memory()
                if valid_memory.size(0) == 0:
                    cr_loss = x.sum() * 0.0
                else:
                    cr_loss = calc_regloss(x, x_aug, valid_memory, valid_memory_ids, batch.subject_id, temperature=args.memory_temperature)
            view_loss = (model.calc_loss(x, x_aug, temperature=args.batch_temperature, sym=args.contrastive_symmetric)
                         + reg_sign * (args.reg_lambda * reg) + args.cr_lambda * cr_loss)
            view_loss_all += view_loss.item() * batch.num_graphs
            keep_prob_all += keep_prob.item()
            sampled_keep_all += sampled_keep.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()

            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)

            if is_paper_exact:
                with torch.no_grad():
                    edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
                    _, edge_mask = sample_ordered_concrete_mask(
                        edge_logits, temperature=args.concrete_temperature)
            else:
                with torch.no_grad():
                    edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
                    rev_idx = compute_reverse_index(batch.edge_index)
                    sym_logits = symmetrize_edge_logits(batch.edge_index, edge_logits, rev_idx)
                    noise = sample_symmetric_logistic_noise(batch.edge_index, rev_idx, bias=bias, device=device)
                    gate_inputs = (noise + sym_logits) / temperature
                    edge_mask = torch.sigmoid(gate_inputs)
            aug_edge_weight = batch.edge_weight * edge_mask

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, aug_edge_weight)

            if is_paper_exact:
                # queue already includes this batch's embeddings, pushed
                # above in the view-update half-step -- no second push here
                memory = paper_memory_bank.get_memory()
                cr_loss = calc_regloss_paper(x, x_aug, memory, temperature=args.memory_temperature)
            else:
                valid_memory, valid_memory_ids = memory_bank.get_valid_memory()
                if valid_memory.size(0) == 0:
                    cr_loss = x.sum() * 0.0
                else:
                    cr_loss = calc_regloss(x, x_aug, valid_memory, valid_memory_ids, batch.subject_id, temperature=args.memory_temperature)

            model_loss = (model.calc_loss(x, x_aug, temperature=args.batch_temperature, sym=args.contrastive_symmetric)
                          + args.cr_lambda * cr_loss)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()

            if not is_paper_exact:
                # corrected profile only -- enqueue last, so this batch never
                # appears as its own negative. paper_exact already pushed
                # once, before the view-update loss, per the literal bug 5
                # behavior; pushing again here would diverge further still.
                if args.feature_type == 'instance':
                    memory_bank.push(x_aug.detach(), batch.subject_id)
                else:
                    memory_bank.push(x_aug.detach().mean(dim=0).unsqueeze(dim=0),
                                      torch.tensor([-1], dtype=torch.long, device=device))


        fin_model_loss = model_loss_all / (len(dataloader) * args.batch_size)
        fin_view_loss = view_loss_all / (len(dataloader) * args.batch_size)
        fin_keep_prob = keep_prob_all / len(dataloader)
        fin_sampled_keep = sampled_keep_all / len(dataloader)

        # explicit names -- 'Reg' used to mean drop-probability in old logs and
        # keep-probability after the paper-literal fix; ambiguous, now removed
        logging.info('Epoch {}, Model Loss {}, View Loss {}, KeepProb {}, DropProb {}, SampledKeep {}'.format(
            epoch, fin_model_loss, fin_view_loss, fin_keep_prob, 1.0 - fin_keep_prob, fin_sampled_keep))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        keep_probs.append(fin_keep_prob)

        should_evaluate = epoch % args.eval_interval == 0 or epoch == args.epochs
        if should_evaluate:
            model.eval()
            # train/val only -- test is never computed during training, so it
            # cannot influence checkpoint selection even by accident
            train_score, val_score, _ = ee.kf_embedding_evaluation(
                model, dataset, fixed_splits, representation=args.eval_representation, flag=True, include_test=False)

            logging.info(
                "Metric: {} Train_mean: {} Val_mean: {}".format(evaluator.eval_metric, train_score[0], val_score[0]))
            logging.info(
                "Metric: {} Train_std: {} Val_std: {}".format(evaluator.eval_metric, train_score[1], val_score[1]))
            logging.info(
                "Metric: f1 Train_mean: {} Val_mean: {}".format(train_score[2], val_score[2]))
            logging.info(
                "Metric: f1 Train_std: {} Val_std: {}".format(train_score[3], val_score[3]))
            logging.info(
                "Metric: sen Train_mean: {} Val_mean: {}".format(train_score[4], val_score[4]))
            logging.info(
                "Metric: sen Train_std: {} Val_std: {}".format(train_score[5], val_score[5]))
            logging.info(
                "Metric: spe Train_mean: {} Val_mean: {}".format(train_score[6], val_score[6]))
            logging.info(
                "Metric: spe Train_std: {} Val_std: {}".format(train_score[7], val_score[7]))
            logging.info(
                "Metric: auc Train_mean: {} Val_mean: {}".format(train_score[8], val_score[8]))
            logging.info(
                "Metric: auc Train_std: {} Val_std: {}".format(train_score[9], val_score[9]))

            if val_score[0] > best_val_accuracy:
                best_val_accuracy = val_score[0]
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'view_state_dict': view_learner.state_dict(),
                    'model_optimizer': model_optimizer.state_dict(),
                    'view_optimizer': view_optimizer.state_dict(),
                    'validation_score': val_score,
                }, args.checkpoint_path)
                logging.info('New best validation accuracy %.4f at epoch %d -- checkpoint saved', best_val_accuracy, epoch)

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: %d (validation accuracy %.4f)', best_epoch, best_val_accuracy)

    # reload the single checkpoint selected on validation, evaluate test once
    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # %-style throughout -- these are lazily formatted by logging via
    # msg % args, so mixing in str.format()'s '{}' placeholders here would
    # raise "not all arguments converted" the moment the line actually fires
    score_fmt = ('acc_mean: %.4f acc_std: %.4f f1_mean: %.4f f1_std: %.4f sen_mean: %.4f sen_std: %.4f '
                 'spe_mean: %.4f spe_std: %.4f auc_mean: %.4f auc_std: %.4f')

    if is_paper_exact:
        # paper_exact: the paper's own reported protocol -- plain (non-
        # stratified) K-fold CV directly on embeddings from the validation-
        # selected checkpoint, no separate held-out test fold beyond that.
        full_loader = DataLoader(dataset, batch_size=128)
        embeddings, labels = get_emb_y(full_loader, model, device, representation=args.eval_representation)
        paper_score = paper_five_fold_evaluation(embeddings, labels, seed=args.seed, n_splits=args.n_folds)
        logging.info(
            'PaperFiveFoldScore (checkpoint epoch %d, plain %d-fold CV on embeddings): ' + score_fmt,
            best_epoch, args.n_folds,
            paper_score['acc_mean'], paper_score['acc_std'],
            paper_score['f1_mean'], paper_score['f1_std'],
            paper_score['sen_mean'], paper_score['sen_std'],
            paper_score['spe_mean'], paper_score['spe_std'],
            paper_score['auc_mean'], paper_score['auc_std'])
    else:
        # fit_on_train_val=True means the returned "val_score" here is an
        # in-sample score (validation was merged into the fit set) -- it is
        # never held out at this point, so it must not be reported or labeled
        # as a validation metric. Only FinalFitScore (train+val, in-sample) and
        # FinalTestScore (the genuinely held-out 20%) are meaningful here.
        fit_score, _, test_score = ee.kf_embedding_evaluation(
            model, dataset, fixed_splits, representation=args.eval_representation, include_test=True, fit_on_train_val=True)

        logging.info('FinalFitScore (checkpoint epoch %d, train+val, in-sample): ' + score_fmt, best_epoch, *fit_score)
        logging.info('FinalTestScore (checkpoint epoch %d, evaluated once, held out): ' + score_fmt, best_epoch, *test_score)

    return best_val_accuracy


def arg_parse():
    parser = argparse.ArgumentParser(description='A-GCL ABIDE')

    parser.add_argument('--training_profile', type=str, default='corrected', choices=['paper_exact', 'corrected'],
                        help='corrected (default): this project\'s verified, tested fixes. paper_exact: literal '
                             'paper formulas (asymmetric mask, zero-init queue, no subject exclusion, '
                             'push-before-loss, collapse-prone regularizer -- matches printed paper, not the '
                             'released/corrected code) -- an explicit opt-in, not what a bare run executes. '
                             'Nothing about the corrected code path is deleted or changed by selecting '
                             'paper_exact -- see docs/changes.md.')
    parser.add_argument('--concrete_temperature', type=float, default=1.0,
                        help='temperature for the Concrete/Gumbel-sigmoid mask reparameterization (both profiles)')
    parser.add_argument('--allow_dataset_mismatch', type=str2bool, default=True,
                        help="paper_exact expects the paper's exact ABIDE-I configuration (987 subjects, 116 ROIs). "
                             "we only have 90-ROI/956-subject data, so this defaults to True (warn, don't block) -- "
                             "set False to hard-require an exact dataset match instead.")
    parser.add_argument('--contrastive_symmetric', type=str2bool, default=True,
                        help='symmetric (both-direction) contrastive loss term (True=current/corrected code) vs. '
                             'the paper-literal one-directional loss (False, sym=False in calc_loss)')

    parser.add_argument('--name', type=str, default='ABIDE',
                        help='dataset.')
    parser.add_argument('--path', type=str, default='',
                        help='path of dataset.')
    parser.add_argument('--template', type=int, default=116,
                        help='dataset template.')
    parser.add_argument('--model_lr', type=float, default=0.0005,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.0005,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=2,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.3,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=2.0,
                        help='View Learner Edge Perturb Regularization Strength')
    parser.add_argument('--regularizer_mode', type=str, default='budget',
                        choices=['budget', 'paper_keep', 'target_keep'],
                        help="budget (default): reg=R(mu)=mean(mu) with sign '+' -- opposes the view "
                             "learner's drop-pressure (AD-GCL perturbation budget), stable keep-probability. "
                             "paper_keep: same reg, sign '-', exactly the paper's literal printed formula -- "
                             "no opposing term, keep-probability collapses monotonically (forced under "
                             "--training_profile paper_exact regardless of this flag). target_keep: "
                             "engineering ablation, NOT the paper's objective -- pulls keep-probability "
                             "toward --target_keep instead")
    parser.add_argument('--target_keep', type=float, default=0.20,
                        help='target keep ratio, only used when --regularizer_mode target_keep')
    parser.add_argument('--normalize_nodes', type=str2bool, default=True,
                        help='L2-normalize node embeddings before pooling (True=current code, False=paper-literal ablation)')
    parser.add_argument('--message_relu', type=str2bool, default=True,
                        help='apply ReLU inside WGINConv.message() (True=current code, False=paper-literal ablation)')
    parser.add_argument('--post_bn_relu', type=str2bool, default=True,
                        help='extra ReLU after BatchNorm between GIN layers (True=current code, False=paper-literal ablation)')
    parser.add_argument('--batch_temperature', type=float, default=0.2,
                        help='InfoNCE temperature for the batch contrastive loss (undocumented in the paper; 1.0 is the paper-aligned candidate)')
    parser.add_argument('--memory_temperature', type=float, default=0.1,
                        help='InfoNCE temperature for the memory-bank loss (undocumented in the paper; 1.0 is the paper-aligned candidate)')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear",
                        help="Downstream classifier is linear or non-linear")
    parser.add_argument('--max_length', type=int, default=256,
                        help='max length of memory bank')
    parser.add_argument('--cr_lambda', type=float, default=0.4,
                        help='Regularization coefficients for loss of cross-batch memory bank')
    parser.add_argument('--memory_type', type=str, default='queue', choices=['momentum', 'queue'],
                        help="type of memory bank")
    parser.add_argument('--feature_type', type=str, default='instance', choices=['mean', 'instance'],
                        help="type of feature in memory bank")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n_folds', type=int, default=5,
                        help='number of stratified CV folds, fixed once at the start of training')
    parser.add_argument('--eval_representation', type=str, default='z', choices=['z', 'h'],
                        help='which embedding to evaluate downstream: z (post-projection-head, paper-primary) or h (pre-projection, ablation)')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/agcl_queue_best.pt',
                        help='where to save the best-validation-accuracy checkpoint')

    args = parser.parse_args()
    # used only by apply_training_profile() to warn when a user-provided
    # flag gets silently clobbered by the paper_exact override table
    args._cli_defaults = {action.dest: action.default for action in parser._actions}
    return args


if __name__ == '__main__':
    args = arg_parse()
    run(args)