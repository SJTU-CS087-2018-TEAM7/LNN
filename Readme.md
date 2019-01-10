# Logic Neural Networks (LNN)

## Steps to Start Training

- run `./dataset/LogicSyn/gen_disjunctive_clauses.py` to generate synthetic dataset
- run `mkdir experiments` to create this directory
- run `mkdir statistics` 
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

- Fully LNN (4 Logic Layers):
- Mixed LNN (2 Logic Layers + 2 Fully Connected Layers): 
- Fully NN (4 Fully Connected Layers):

### Influence of Regularization

#### Results on Synthetic Dataset

The statistics of dataset remains the same as previous.

- Without regularization, the histogram of variables in logic kernels:

### Interpretability

### Results on Synthetic Dataset

The statistics of dataset:

<<<<<<< HEAD
```
input_len = 10
disjunction_maxlen = 3
disjunction_num = 3
```
=======
use `tensorboard --logdir statistics` to examine the histograms of selectors and negators

>>>>>>> 5634ca30857fd822ca927987215be98d9010029d
