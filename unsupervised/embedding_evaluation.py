import numpy as np
from sklearn.utils import shuffle
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

from numpy import interp

import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score
import time

def create_fixed_splits(dataset, n_splits=5, seed=123):
	"""Precompute stratified train/val/test indices once, so every evaluation
	across every epoch reuses the exact same folds instead of resampling."""
	labels = np.array([int(dataset[i].y.item()) for i in range(len(dataset))])

	outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

	splits = []
	for fold_id, (train_val_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(labels)), labels)):
		train_idx, val_idx = train_test_split(
			train_val_idx,
			test_size=0.20,
			stratify=labels[train_val_idx],
			random_state=seed + fold_id
		)
		splits.append({"train": train_idx, "val": val_idx, "test": test_idx})

	return splits


def paper_five_fold_evaluation(embeddings, labels, seed=123, n_splits=5):
	"""paper_exact profile: the paper's own reported evaluation protocol --
	plain (non-stratified) K-Fold cross-validation directly on the encoder's
	embeddings, one held-out fold scored per split, no separate val/test
	split and no checkpoint-based epoch selection beyond the single
	validation-selected checkpoint already used to produce `embeddings`.
	Kept alongside, not in place of, the corrected kf_embedding_evaluation
	(stratified fixed splits, val-only checkpoint selection, held-out test
	fold, fit-on-train+val refit)."""
	labels = np.ravel(labels)
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

	accs, sens, spes, f1s, aucs = [], [], [], [], []

	for train_idx, test_idx in kf.split(embeddings):
		train_emb, test_emb = embeddings[train_idx], embeddings[test_idx]
		train_y, test_y = labels[train_idx], labels[test_idx]

		pipeline = Pipeline([('scaler', StandardScaler()),
		                      ('clf', LinearSVC(dual=False, fit_intercept=True, max_iter=10000))])
		params_dict = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
		classifier = GridSearchCV(pipeline, params_dict, cv=5, scoring='accuracy', n_jobs=16, verbose=0)
		classifier.fit(train_emb, train_y)

		pred = classifier.predict(test_emb)

		accs.append(accuracy_score(test_y, pred))
		sens.append(recall_score(test_y, pred, pos_label=1))
		spes.append(recall_score(test_y, pred, pos_label=0))
		f1s.append(f1_score(test_y, pred))

		dec = classifier.decision_function(test_emb)
		try:
			aucs.append(roc_auc_score(test_y, dec))
		except ValueError:
			aucs.append(float('nan'))

	return {
		'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
		'sen_mean': np.mean(sens), 'sen_std': np.std(sens),
		'spe_mean': np.mean(spes), 'spe_std': np.std(spes),
		'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
		'auc_mean': np.nanmean(aucs), 'auc_std': np.nanstd(aucs),
	}


def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False, representation='z'):
	# train_emb, train_y
	x, y = encoder.get_embeddings(loader, device, representation=representation, is_rand_label=is_rand_label)

	if dtype == 'numpy':
		return x,y
	elif dtype == 'torch':
		return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
	else:
		raise NotImplementedError

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def sensitivity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative
	Condition_negative = tp_sum + fn_sum + 1e-6
	sensitivity = tp_sum / Condition_negative

	return sensitivity

def specificity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative

	Condition_negative = tn_sum + fp_sum + 1e-6
	Specificity = tn_sum / Condition_negative

	return Specificity


class EmbeddingEvaluation():
	def __init__(self, base_classifier, evaluator, task_type, num_tasks, device, params_dict=None, param_search=True,is_rand_label=False):
		self.is_rand_label = is_rand_label
		self.base_classifier = base_classifier
		self.evaluator = evaluator
		self.eval_metric = evaluator.eval_metric
		self.task_type = task_type
		self.num_tasks = num_tasks
		self.device = device
		self.param_search = param_search
		self.params_dict = params_dict
		if self.eval_metric == 'rmse':
			self.gscv_scoring_name = 'neg_root_mean_squared_error'
		elif self.eval_metric == 'mae':
			self.gscv_scoring_name = 'neg_mean_absolute_error'
		elif self.eval_metric == 'rocauc':
			self.gscv_scoring_name = 'roc_auc'
		elif self.eval_metric == 'accuracy':
			self.gscv_scoring_name = 'accuracy'
		else:
			raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

		self.classifier = None
	def scorer(self, y_true, y_raw):

		input_dict = {"y_true": y_true, "y_pred": y_raw}
		score = self.evaluator.eval(input_dict)[self.eval_metric]
		return score

	def ee_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
		if self.param_search:
			# scaler must live INSIDE the searched pipeline, so each inner CV
			# fold fits its own scaler -- fitting it outside leaks validation-
			# fold statistics into the scaling used to score that fold.
			inner_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', self.base_classifier)])
			params_dict = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
			self.classifier = GridSearchCV(inner_pipeline, params_dict, cv=5, scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
		else:
			self.classifier = make_pipeline(StandardScaler(), self.base_classifier)

		if np.isnan(train_emb).any():
			print("Has NaNs ... ignoring them")
			train_emb = np.nan_to_num(train_emb)
		
		if np.isnan(val_emb).any():
			print("Has NaNs ... ignoring them")
			val_emb = np.nan_to_num(val_emb)
		if np.isnan(test_emb).any():
			print("Has NaNs ... ignoring them")
			test_emb = np.nan_to_num(test_emb)
			
		self.classifier.fit(train_emb, np.squeeze(train_y))

		if self.eval_metric == 'accuracy':
			train_raw = self.classifier.predict(train_emb)
			val_raw = self.classifier.predict(val_emb)
			test_raw = self.classifier.predict(test_emb)
		else:
			train_raw = self.classifier.predict_proba(train_emb)[:, 1]
			val_raw = self.classifier.predict_proba(val_emb)[:, 1]
			test_raw = self.classifier.predict_proba(test_emb)[:, 1]

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

	def ee_multioutput_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):

		params_dict = {
			'multioutputclassifier__estimator__C': [1e-1, 1e0, 1e1, 1e2]}
		self.classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(
			self.base_classifier, n_jobs=-1))
		
		if np.isnan(train_y).any():
			print("Has NaNs ... ignoring them")
			train_y = np.nan_to_num(train_y)
		self.classifier.fit(train_emb, train_y)

		train_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(train_emb)])
		val_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(val_emb)])
		test_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(test_emb)])

		return train_raw, val_raw, test_raw

	def ee_regression(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
		if self.param_search:
			params_dict = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
# 			params_dict = {'alpha': [500, 50, 5, 0.5, 0.05, 0.005, 0.0005]}
			self.classifier = GridSearchCV(self.base_classifier, params_dict, cv=5,
			                          scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
		else:
			self.classifier = self.base_classifier

		self.classifier.fit(train_emb, np.squeeze(train_y))

		train_raw = self.classifier.predict(train_emb)
		val_raw = self.classifier.predict(val_emb)
		test_raw = self.classifier.predict(test_emb)

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

	def embedding_evaluation(self, encoder, train_loader, valid_loader, test_loader, flag, representation='z', fit_on_train_val=False):
		encoder.eval()
		val_start = time.time()
		train_emb, train_y = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label, representation=representation)
		val_emb, val_y = get_emb_y(valid_loader, encoder, self.device, is_rand_label=self.is_rand_label, representation=representation)
		test_emb, test_y = get_emb_y(test_loader, encoder, self.device, is_rand_label=self.is_rand_label, representation=representation)
		val_end = time.time()
		running_time = val_end-val_start
		if flag:
			print('validation time cost : %.5f sec' %running_time)

		if fit_on_train_val:
			# final evaluation only: the checkpoint was already selected via
			# validation, so refit on the complete outer 80% (train+val)
			# before scoring the held-out 20% test fold, instead of leaving
			# 16% of the data unused for the number that gets reported.
			# train_score/sen/spe/f1/auc below then reflect this combined
			# fit set, not the original train-only split -- val is no longer
			# a meaningful held-out signal once merged in, which is fine
			# since it isn't used for any selection at this point.
			train_emb = np.concatenate([train_emb, val_emb], axis=0)
			train_y = np.concatenate([train_y, val_y], axis=0)

		if 'classification' in self.task_type:

			if self.num_tasks == 1:
				train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, train_y, val_emb, val_y, test_emb,
				                                                        test_y)
			elif self.num_tasks > 1:
				train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y,
				                                                                    test_emb, test_y)
			else:
				raise NotImplementedError
		else:
			if self.num_tasks == 1:
				train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
			else:
				raise NotImplementedError
		

		train_score = self.scorer(train_y, train_raw)
		val_score = self.scorer(val_y, val_raw)
		test_score = self.scorer(test_y, test_raw)

		train_sen_score = sensitivity(train_raw, train_y)
		val_sen_score = sensitivity(val_raw, val_y)
		test_sen_score = sensitivity(test_raw, test_y)

		train_spe_score = specificity(train_raw, train_y)
		val_spe_score = specificity(val_raw, val_y)
		test_spe_score = specificity(test_raw, test_y)

		train_f1_score = f1_score(train_y, train_raw)
		val_f1_score = f1_score(val_y, val_raw)
		test_f1_score = f1_score(test_y, test_raw)

		if hasattr(self.classifier, 'decision_function'):
			train_dec = self.classifier.decision_function(train_emb)
			val_dec = self.classifier.decision_function(val_emb)
			test_dec = self.classifier.decision_function(test_emb)
			try:
				train_auc = roc_auc_score(np.ravel(train_y), train_dec)
				val_auc = roc_auc_score(np.ravel(val_y), val_dec)
				test_auc = roc_auc_score(np.ravel(test_y), test_dec)
			except ValueError:
				train_auc = val_auc = test_auc = float('nan')
		else:
			train_auc = val_auc = test_auc = float('nan')

		return (train_score, val_score, test_score, train_f1_score, val_f1_score, test_f1_score,
				train_sen_score, val_sen_score, test_sen_score, train_spe_score, val_spe_score, test_spe_score,
				train_auc, val_auc, test_auc, running_time)

	def kf_embedding_evaluation(self, encoder, dataset, fixed_splits, representation='z', batch_size=128, flag=False, include_test=True, fit_on_train_val=False):
		kf_train = []
		kf_val = []
		kf_test = []
		kf_train_f1 = []
		kf_val_f1 = []
		kf_test_f1 = []
		kf_train_sen = []
		kf_val_sen = []
		kf_test_sen = []
		kf_train_spe = []
		kf_val_spe = []
		kf_test_spe = []
		kf_train_auc = []
		kf_val_auc = []
		kf_test_auc = []
		running_times = []

		for split in fixed_splits:
			train_dataset = [dataset[int(i)] for i in split["train"]]
			val_dataset = [dataset[int(i)] for i in split["val"]]
			# same-shaped test data every time -- either the real fixed test
			# fold, or the val fold again (unused, just keeps the plumbing
			# below identical) when include_test=False for training-time evals
			test_dataset = [dataset[int(i)] for i in split["test"]] if include_test else val_dataset

			train_loader = DataLoader(train_dataset, batch_size=batch_size)
			valid_loader = DataLoader(val_dataset, batch_size=batch_size)
			test_loader = DataLoader(test_dataset, batch_size=batch_size)

			# embedding_evaluation -> get_emb_y -> encoder.get_embeddings -> forward
			(train_score, val_score, test_score,
			 train_f1, val_f1, test_f1,
			 train_sen, val_sen, test_sen,
			 train_spe, val_spe, test_spe,
			 train_auc, val_auc, test_auc,
			 running_time) = self.embedding_evaluation(
				encoder, train_loader, valid_loader, test_loader, flag, representation=representation, fit_on_train_val=fit_on_train_val)

			if not include_test:
				test_score = test_f1 = test_sen = test_spe = test_auc = float('nan')

			running_times.append(running_time)

			kf_train_f1.append(train_f1)
			kf_val_f1.append(val_f1)
			kf_test_f1.append(test_f1)

			kf_train_spe.append(train_spe)
			kf_val_spe.append(val_spe)
			kf_test_spe.append(test_spe)

			kf_train.append(train_score)
			kf_val.append(val_score)
			kf_test.append(test_score)

			kf_train_sen.append(train_sen)
			kf_val_sen.append(val_sen)
			kf_test_sen.append(test_sen)

			kf_train_auc.append(train_auc)
			kf_val_auc.append(val_auc)
			kf_test_auc.append(test_auc)

		mean_time = np.array(running_times).mean()
		print("mean validation time %.5f:\n"% mean_time)

		kf_train_ms = [np.array(kf_train).mean(), np.array(kf_train).std(), np.array(kf_train_f1).mean(),
						np.array(kf_train_f1).std(),
						np.array(kf_train_sen).mean(), np.array(kf_train_sen).std(), np.array(kf_train_spe).mean(),
						np.array(kf_train_spe).std(),
						np.nanmean(kf_train_auc), np.nanstd(kf_train_auc)]
		kf_val_ms = [np.array(kf_val).mean(), np.array(kf_val).std(), np.array(kf_val_f1).mean(),
						np.array(kf_val_f1).std(),
						np.array(kf_val_sen).mean(), np.array(kf_val_sen).std(), np.array(kf_val_spe).mean(),
						np.array(kf_val_spe).std(),
						np.nanmean(kf_val_auc), np.nanstd(kf_val_auc)]
		kf_test_ms = [np.array(kf_test).mean(), np.array(kf_test).std(), np.array(kf_test_f1).mean(),
						np.array(kf_test_f1).std(),
						np.array(kf_test_sen).mean(), np.array(kf_test_sen).std(), np.array(kf_test_spe).mean(),
						np.array(kf_test_spe).std(),
						np.nanmean(kf_test_auc), np.nanstd(kf_test_auc)]

		return kf_train_ms, kf_val_ms, kf_test_ms

