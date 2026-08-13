# Final Consolidated Summary of Issues and Corrections

## 1. Incorrect augmented graph passed to the main model

* Previously, the augmented view used the learned mask as the edge weight, effectively replacing the original functional connectivity values.
* The correct formulation from the paper is (E_{\text{aug}} = E_{\text{original}} \odot B).
* Edge logits may initially have shape `[batch_edges, 1]`; after squeezing, mask (B) has shape `[batch_edges]`, matching the original edge weights.
* Both original positive and negative FC values are now retained, while the mask only controls their strength or removal.
* This ensures that the model learns from the masked FC graph rather than from an artificial mask-only graph.
* Files: `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`, `tests/test_augmentation.py`.

## 2. Incorrect and ambiguous augmentation regularization

* The old variable named `Reg` did not clearly distinguish edge-keeping probability from edge-dropping probability.
* The implementation now defines (\mu=\sigma(\text{edge logits})) and uses (R(\mu)=\text{mean}(\mu)), as specified in the paper.
* The view learner maximizes contrastive difficulty while minimizing the keep probability through the correctly signed adversarial objective.
* Logs now separately report `KeepProb`, `DropProb`, and `SampledKeep`, allowing mask collapse to be detected.
* An optional `target_keep` mode was added as an engineering ablation, but it is not presented as the paper’s original objective.
* Files: `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`.

## 3. Asymmetric FC edge masks

* An undirected FC connection was stored as two directed edges, ((u,v)) and ((v,u)).
* Previously, both directions could receive different logits and different random noise, producing an asymmetric augmented FC matrix.
* Reverse-edge indices are now calculated, and directional logits are averaged before mask sampling.
* Both directions also receive the same Logistic-Concrete noise, ensuring (B_{uv}=B_{vu}).
* Self-loops are handled independently, and edges belonging to different subjects cannot be paired accidentally.
* Files: `unsupervised/view_learner.py`, `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`, `tests/test_augmentation.py`.

## 4. Incorrect memory-bank behaviour

* The old queue contained uninitialized zero rows and could expose the current batch as its own memory negative.
* The revised queue tracks valid positions, stores detached projected embeddings, and records stable subject IDs.
* For batch size 32, `x` and `x_aug` have shape `[32, 32]`, while valid memory gradually grows to `[256, 32]`.
* Each anchor has its own augmented embedding as the positive; batch negatives are handled by the batch loss and past-subject negatives by the memory loss.
* Same-subject historical embeddings and anchors without valid negatives are now excluded before `logsumexp`, preventing false negatives and NaN gradients.
* Files: `agcl_ABIDE_queue.py`, `datasets/abideDataset.py`, `tests/test_memory_bank.py`.

## 5. Random and non-stratified fold generation

* Previously, folds were regenerated during different evaluation epochs, so accuracy changes mixed model improvement with split variation.
* The code now creates five stratified outer folds once using a fixed seed and reuses them throughout training.
* Each fold initially contains approximately 64% training, 16% validation, and 20% testing subjects.
* Training-time evaluation uses the 64% training and 16% validation subsets, while the outer test indices remain unused.
* After checkpoint selection, the final SVM is refitted on train plus validation, giving 80% fitting data and 20% held-out test data.
* Files: `unsupervised/embedding_evaluation.py`, `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`, `tests/test_fixed_splits.py`.

## 6. Test-based best-epoch selection

* The previous implementation evaluated the test folds repeatedly and selected different best epochs for accuracy, F1, sensitivity, specificity, and AUC.
* This leaked test information into model selection and created a result that did not correspond to one actual model checkpoint.
* Epoch selection now uses only mean validation accuracy from the fixed folds.
* An epoch-0 checkpoint is saved, and the final epoch is always evaluated, preventing missing or stale checkpoints.
* The selected checkpoint is reloaded once, and all final test metrics are calculated from that same checkpoint.
* Files: `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`, `unsupervised/embedding_evaluation.py`, `tests/test_checkpoint_selection.py`.

## 7. Evaluating the wrong embedding

* Previously, evaluation passed only `model.encoder`, so the projection head was completely bypassed.
* The paper defines (z(G)) after the GIN encoder, global-add pooling, and the two-layer projection head.
* The full `GInfoMinMax` model is now passed to embedding evaluation, and projected `z` is the primary representation.
* The pre-projection representation `h` remains available through `--eval_representation h` as an ablation.
* The scaler is also placed inside the inner `GridSearchCV`, preventing feature-scaling leakage between inner folds.
* Files: `unsupervised/learning/ginfominmax.py`, `unsupervised/embedding_evaluation.py`, `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`, `tests/test_embeddings.py`.

## 8. Hidden deviations from the paper’s GIN architecture

* The original encoder applied ReLU inside message passing, normalized node embeddings before pooling, and added a post-BatchNorm ReLU.
* These operations are not present in the literal GIN equations provided in the paper and can change signed FC message propagation.
* Separate switches were added for `message_relu`, `normalize_nodes`, and `post_bn_relu`.
* Batch and memory contrastive temperatures were also exposed instead of remaining undocumented constants.
* This allows controlled comparison between the original repository configuration and the paper-literal configuration.
* Files: `unsupervised/encoder/tu_encoder.py`, `unsupervised/convs/wgin_conv.py`, `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`.

## Supporting updates

* Added 32 tests covering augmentation, symmetry, memory, gradients, fixed splits, checkpointing and embedding selection.
* Added `pytest` to `requirements.txt`.
* Preserved subject-to-feature filename matching and graph normalization corrections in `datasets/abideDataset.py`.
* Added explicit final outputs: `FinalFitScore` for the in-sample 80% fit and `FinalTestScore` for the held-out 20%.
* Static compilation and diff validation passed.

## Remaining operational action

The code supports the paper-literal configuration, but the current `scripts/train_queue_200ep_cpu.slurm` does not pass those options. Add:

```bash
--normalize_nodes false \
--message_relu false \
--post_bn_relu false \
--batch_temperature 1.0 \
--memory_temperature 1.0 \
--eval_representation z \
--regularizer_mode paper_keep
```

These corrections remove the identified implementation errors, but they do not guarantee 80% accuracy. That claim requires passing all tests and producing a fresh, complete training log from the corrected implementation.
