# Logic Neural Networks (LNN)

## Steps to Start Training

- run `python./dataset/LogicSyn/gen_disjunctive_clauses.py` to generate synthetic dataset
- run `mkdir experiments` to create this directory
- run main.py

use `tensorboard --logdir statistics` to examine the histograms of selectors and negators

## Experiments

### Comparison Between LNN and Conventional NN

#### Results on Synthetic Dataset

The statistics of dataset:

```
input_len = 100
disjunction_maxlen = 50
disjunction_num = 40

train set size: 8000
        positive samples: 3137
        negative samples: 4863
        positive ratio: 0.392125
test set size: 2000
        positive samples: 814
        negative samples: 1186
        positive ratio: 0.407
```

- Fully LNN (4 Logic Layers): `train_loss:0.418 train_acc:0.897 train_auc:0.915 valid_loss:0.417 valid_acc:0.897 valid_auc:0.914`
- Mixed LNN (2 Logic Layers + 2 Fully Connected Layers): `train_loss:0.060 train_acc:0.979 train_auc:0.997 valid_loss:0.075 valid_acc:0.980 valid_auc:0.991`
- Fully NN (4 Fully Connected Layers): `train_loss:-0.000 train_acc:1.000 train_auc:1.000 valid_loss:0.694 valid_acc:0.933 valid_auc:0.972`

### Influence of Regularization

#### Results on Synthetic Dataset

The statistics of dataset remains the same as previous.

- Without regularization, the histogram of variables in logic kernels:

### Interpretability

### Results on Synthetic Dataset

The statistics of dataset:

```
input_len = 10
disjunction_maxlen = 3
disjunction_num = 3

train set: 800
        positive samples: 431
        negative samples: 369
        positive ratio: 0.53875
test size: 200
        positive samples: 103
        negative samples: 97
        positive ratio: 0.515
```
