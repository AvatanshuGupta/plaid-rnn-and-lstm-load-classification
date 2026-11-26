# Load classification: sequential model report

## 1. Setup

| item | value |
|---|---|
| features file | features/all_features.csv |
| target | load |
| classes | 4 |
| records | 1074 |
| windows per record (min/median/max) | 60 / 180 / 806 |
| feature count used | 53 |
| excluded columns | f0_Hz |
| sequence length (windows) | 30 |
| chunks train/val/test | 18533 / 1091 / 1122 |
| architecture | 2-layer, hidden=64, dropout=0.3, bidirectional=False |
| optimiser | AdamW lr=0.002 wd=0.001, batch=64, label-smoothing 0.05, class-weighted CE, grad-clip 1.0, input noise 0.05 |
| early stopping | val loss, patience 12 |
| seed | 42 |
| device | cpu |


## 2. Data split (by record id, stratified)

| class | train | val | test |
|---|---|---|---|
| I | 134 | 28 | 29 |
| R | 213 | 46 | 46 |
| RC | 243 | 52 | 52 |
| RLC | 161 | 35 | 35 |
| **total records** | 751 | 161 | 162 |


## 3. Headline comparison (test set, record-level)

| metric | RNN | RandomForest (baseline) |
|---|---|---|
| accuracy | 0.9877 | 0.9444 |
| balanced_accuracy | 0.9828 | 0.9362 |
| macro_f1 | 0.9849 | 0.9344 |
| weighted_f1 | 0.9875 | 0.9454 |
| macro_precision | 0.9877 | 0.9383 |
| macro_recall | 0.9828 | 0.9362 |
| cohen_kappa | 0.9832 | 0.9248 |
| mcc | 0.9834 | 0.9260 |
| log_loss | 0.0985 | 0.1596 |
| top3_accuracy | 1.0000 | 1.0000 |
| roc_auc_macro_ovr | 0.9969 | 0.9953 |
| accuracy 95% CI | [0.969, 1.000] | [0.907, 0.975] |
| macro-F1 95% CI | [0.960, 1.000] | [0.891, 0.969] |


## 4. Training summary

| model | params | epochs run | best epoch | train time (s) | chunk-level test acc | chunk-level test macro-F1 |
|---|---|---|---|---|---|---|
| RNN | 24452 | 28 | 16 | 125.0 | 0.9724 | 0.9687 |
| RandomForest (baseline) | - | - | - | 3.3 | nan | nan |


![curves](curves.png)


## 5. Detail: RNN

| class | precision | recall | f1 | support | AUC (OvR) |
|---|---|---|---|---|---|
| I | 1.0000 | 0.9310 | 0.9643 | 29 | 0.9904 |
| R | 0.9787 | 1.0000 | 0.9892 | 46 | 0.9974 |
| RC | 1.0000 | 1.0000 | 1.0000 | 52 | 1.0000 |
| RLC | 0.9722 | 1.0000 | 0.9859 | 35 | 1.0000 |


Most frequent confusions (true -> predicted, count):

| true | predicted | count |
|---|---|---|
| I | R | 1 |
| I | RLC | 1 |


![cm](cm_RNN.png)


## 5. Detail: RandomForest (baseline)

| class | precision | recall | f1 | support | AUC (OvR) |
|---|---|---|---|---|---|
| I | 0.7941 | 0.9310 | 0.8571 | 29 | 0.9886 |
| R | 0.9778 | 0.9565 | 0.9670 | 46 | 0.9991 |
| RC | 0.9811 | 1.0000 | 0.9905 | 52 | 1.0000 |
| RLC | 1.0000 | 0.8571 | 0.9231 | 35 | 0.9937 |


Most frequent confusions (true -> predicted, count):

| true | predicted | count |
|---|---|---|
| RLC | I | 5 |
| R | I | 2 |
| I | R | 1 |
| I | RC | 1 |


![f1](per_class_f1.png)


## 6. How to read this

- Headline numbers are **record-level** (chunk probabilities averaged per record), computed on records never seen in training.
- Macro-F1 / balanced accuracy matter more than accuracy if classes are imbalanced.
- If the test set is small the CIs above are wide; differences between models inside the CI are not significant.
- The RandomForest row uses per-record mean/std/min/max of the same features; if the RNN/LSTM does not beat it, the temporal structure is not adding much.
- PLAID has several records per physical appliance; a split by id can still put the same appliance in train and test, so real-world generalisation may be lower.

## 7. SHAP feature attribution (GradientExplainer)
SHAP values are in logit units, computed on standardised features. Importance = |sum of SHAP over the sequence's time steps| averaged over explained test chunks and classes. Plots and full tables are in `shap_<model>/`.

### RNN (additivity error 1.1974 logits)
| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | sin_phi | 0.4879 |
| 2 | i_h3_ratio | 0.3642 |
| 3 | phase_deg | 0.3591 |
| 4 | i_harm_phase_h3_deg | 0.2628 |
| 5 | p_inst_min | 0.2579 |
| 6 | i_h5_ratio | 0.1836 |
| 7 | v_mean | 0.1658 |
| 8 | i_odd_even_ratio | 0.1459 |
| 9 | Q1_var | 0.1429 |
| 10 | B1_siemens | 0.1395 |

![global](shap_rnn/global_importance.png)

![heatmap](shap_rnn/class_feature_heatmap.png)

![temporal](shap_rnn/temporal_importance.png)
