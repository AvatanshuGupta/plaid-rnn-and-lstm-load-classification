# Load classification: sequential model report

## 1. Setup

| item | value |
|---|---|
| features file | features/all_features.csv |
| target | load |
| classes | 3 |
| records | 1074 |
| windows per record (min/median/max) | 60 / 180 / 806 |
| feature count used | 53 |
| excluded columns | f0_Hz |
| sequence length (windows) | 30 |
| chunks train/val/test | 18916 / 1067 / 1060 |
| architecture | 2-layer, hidden=64, dropout=0.3, bidirectional=False |
| optimiser | AdamW lr=0.002 wd=0.001, batch=64, label-smoothing 0.05, class-weighted CE, grad-clip 1.0, input noise 0.05 |
| early stopping | val loss, patience 12 |
| seed | 42 |
| device | cpu |


## 2. Data split (by record id, stratified)

| class | train | val | test |
|---|---|---|---|
| I | 134 | 28 | 29 |
| NL | 404 | 87 | 87 |
| R | 213 | 46 | 46 |
| **total records** | 751 | 161 | 162 |


## 3. Headline comparison (test set, record-level)

| metric | RNN | LSTM | RandomForest (baseline) |
|---|---|---|---|
| accuracy | 0.9630 | 0.9568 | 0.9753 |
| balanced_accuracy | 0.9693 | 0.9579 | 0.9693 |
| macro_f1 | 0.9557 | 0.9495 | 0.9696 |
| weighted_f1 | 0.9634 | 0.9569 | 0.9752 |
| macro_precision | 0.9444 | 0.9422 | 0.9703 |
| macro_recall | 0.9693 | 0.9579 | 0.9693 |
| cohen_kappa | 0.9392 | 0.9287 | 0.9588 |
| mcc | 0.9404 | 0.9295 | 0.9590 |
| log_loss | 0.1710 | 0.1897 | 0.1486 |
| top3_accuracy | - | - | - |
| roc_auc_macro_ovr | 0.9870 | 0.9861 | 0.9952 |
| accuracy 95% CI | [0.932, 0.988] | [0.920, 0.988] | [0.951, 0.994] |
| macro-F1 95% CI | [0.918, 0.987] | [0.906, 0.982] | [0.936, 0.995] |


## 4. Training summary

| model | params | epochs run | best epoch | train time (s) | chunk-level test acc | chunk-level test macro-F1 |
|---|---|---|---|---|---|---|
| RNN | 24387 | 34 | 22 | 342.4 | 0.9509 | 0.9411 |
| LSTM | 72195 | 26 | 14 | 289.9 | 0.9453 | 0.9356 |
| RandomForest (baseline) | - | - | - | 5.6 | nan | nan |


![curves](curves.png)


## 5. Detail: RNN

| class | precision | recall | f1 | support | AUC (OvR) |
|---|---|---|---|---|---|
| I | 0.8750 | 0.9655 | 0.9180 | 29 | 0.9743 |
| NL | 1.0000 | 0.9425 | 0.9704 | 87 | 0.9917 |
| R | 0.9583 | 1.0000 | 0.9787 | 46 | 0.9949 |


Most frequent confusions (true -> predicted, count):

| true | predicted | count |
|---|---|---|
| NL | I | 4 |
| I | R | 1 |
| NL | R | 1 |


![cm](cm_RNN.png)


## 5. Detail: LSTM

| class | precision | recall | f1 | support | AUC (OvR) |
|---|---|---|---|---|---|
| I | 0.9000 | 0.9310 | 0.9153 | 29 | 0.9712 |
| NL | 0.9880 | 0.9425 | 0.9647 | 87 | 0.9911 |
| R | 0.9388 | 1.0000 | 0.9684 | 46 | 0.9959 |


Most frequent confusions (true -> predicted, count):

| true | predicted | count |
|---|---|---|
| NL | I | 3 |
| NL | R | 2 |
| I | NL | 1 |
| I | R | 1 |


![cm](cm_LSTM.png)


## 5. Detail: RandomForest (baseline)

| class | precision | recall | f1 | support | AUC (OvR) |
|---|---|---|---|---|---|
| I | 0.9643 | 0.9310 | 0.9474 | 29 | 0.9984 |
| NL | 0.9884 | 0.9770 | 0.9827 | 87 | 0.9926 |
| R | 0.9583 | 1.0000 | 0.9787 | 46 | 0.9947 |


Most frequent confusions (true -> predicted, count):

| true | predicted | count |
|---|---|---|
| I | NL | 1 |
| I | R | 1 |
| NL | I | 1 |
| NL | R | 1 |


![f1](per_class_f1.png)


## 6. How to read this

- Headline numbers are **record-level** (chunk probabilities averaged per record), computed on records never seen in training.
- Macro-F1 / balanced accuracy matter more than accuracy if classes are imbalanced.
- If the test set is small the CIs above are wide; differences between models inside the CI are not significant.
- The RandomForest row uses per-record mean/std/min/max of the same features; if the RNN/LSTM does not beat it, the temporal structure is not adding much.
- PLAID has several records per physical appliance; a split by id can still put the same appliance in train and test, so real-world generalisation may be lower.
