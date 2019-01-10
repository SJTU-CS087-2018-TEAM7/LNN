import numpy as np
import os
import json


#
#  config parameters
#
samples = 10000         #  the number of samples to generate
train_ratio = 0.8       #  sample * train_ratio samples will be in train.svm
input_len = 100         #  the dimension of input
disjunction_maxlen = 50 #  the maximum length of the disjunction
disjunction_num = 40    #  the number of disjunctions in the disjunction
data_dir = os.path.dirname(os.path.abspath(__file__))

def calculate(X, disjunctions):
    for disjunction in disjunctions:
        disjunction_true = False
        for p in disjunction:
            negate = p < 0
            p = abs(p)-1
            if (X[p] == 1 and not negate) or (X[p] == 0 and negate):
                disjunction_true = True
                break
        if not disjunction_true:
            return 0
    return 1


def write_file(X, y, filename):
    with open(filename, "w") as f:
        num_lines = X.shape[0]
        for i in range(num_lines):
            f.write(str(y[i]) + " " + " ".join([str(v) for v in X[i].tolist()]) + "\n")


def write_meta():
    meta = {'field_sizes': [1 for i in range(input_len)]}
    with open(os.path.join(data_dir, 'meta.txt'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)

def write_rule(disjunctions):
    with open(os.path.join(data_dir, 'rule.txt'), 'w') as f:
        for disjunction in disjunctions:
            for p in disjunction:
                f.write('%s ' % str(p))
            f.write('\n')

def main():
    disjunctions = []
    for _ in range(disjunction_num):
        disjunction_len = np.random.randint(1, disjunction_maxlen+1)
        disjunction = np.random.choice(range(1, input_len+1), size=disjunction_len, replace=False)
        disjunction = sorted(disjunction)
        for i in range(len(disjunction)):
            if np.random.randint(2) == 0:
                disjunction[i] = -disjunction[i]
        disjunctions.append(disjunction)

    X_all = np.random.randint(2, size=[samples, input_len], dtype=np.int32)
    y_all = np.zeros([samples], dtype=np.int32)

    for i in range(samples):
        y_all[i] = calculate(X_all[i], disjunctions)

    train_num = int(samples * train_ratio)

    write_rule(disjunctions)
    write_file(X_all[:train_num], y_all[:train_num], os.path.join(data_dir, 'raw/train.svm'))
    write_file(X_all[train_num:], y_all[train_num:], os.path.join(data_dir, 'raw/test.svm'))
    write_meta()


def mkdirs():
    raw_path = os.path.join(data_dir, "raw")
    hdf_path = os.path.join(data_dir, "hdf")
    feature_path = os.path.join(data_dir, "feature")
    dirs = [raw_path, hdf_path, feature_path]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

mkdirs()
main()