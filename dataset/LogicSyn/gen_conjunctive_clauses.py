import numpy as np
import os
import json


#
#  config parameters
#
samples = 10000         #  the number of samples to generate
train_ratio = 0.8       #  sample * train_ratio samples will be in train.svm
input_len = 100         #  the dimension of input
conjunction_maxlen = 50 #  the maximum length of the conjunction
conjunction_num = 40    #  the number of conjunctions in the disjunction
data_dir = os.path.dirname(os.path.abspath(__file__))

def calculate(X, conjunctions):
    for conjunction in conjunctions:
        conjunction_true = True
        for p in conjunction:
            negate = p < 0
            p = abs(p)-1
            if (X[p] == 1 and negate) or (X[p] == 0 and not negate):
                conjunction_true = False
                break
        if conjunction_true:
            return 1
    return 0


def write_file(X, y, filename):
    with open(filename, "w") as f:
        num_lines = X.shape[0]
        for i in range(num_lines):
            f.write(str(y[i]) + " " + " ".join([str(v) for v in X[i].tolist()]) + "\n")


def write_meta():
    meta = {'field_sizes': [1 for i in range(input_len)]}
    with open(os.path.join(data_dir, 'meta.txt'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main():
    conjunctions = []
    for _ in range(conjunction_num):
        conjunction_len = np.random.randint(1, conjunction_maxlen+1)
        conjunction = np.random.choice(range(1, input_len+1), size=conjunction_len, replace=False)
        conjunction = sorted(conjunction)
        for i in range(len(conjunction)):
            if np.random.randint(2) == 0:
                conjunction[i] = -conjunction[i]
        conjunctions.append(conjunction)

    X_all = np.random.randint(2, size=[samples, input_len], dtype=np.int32)
    y_all = np.zeros([samples], dtype=np.int32)

    for i in range(samples):
        y_all[i] = calculate(X_all[i], conjunctions)

    train_num = int(samples * train_ratio)

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