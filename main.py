import time
import sys
import json
import argparse
import matplotlib.pyplot as plt
from dataset import as_dataset
from utils import MyEncoder

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='LNN')
# environment
parser.add_argument('--run_name',       type=str,   default='default_run_name')
parser.add_argument('--run_dir',        type=str,   default='experiments/')
# train
parser.add_argument('--batch_size',     type=int,   default=32)
parser.add_argument('--eval_per_steps', type=int,   default=30)
parser.add_argument('--max_rounds',     type=int,   default=100)
parser.add_argument('--lr',             type=float, default=0.0001)
parser.add_argument('--patience',       type=int,   default=5)
# data
parser.add_argument('--use_ratio',      type=float, default=0.1)
parser.add_argument('--train_ratio',    type=float, default=0.8)
# model
args = parser.parse_args()

args.run_name = args.run_dir + args.run_name
with open(args.run_name + '.args', 'w') as f:
    f.write(str(args))

#################################################################

def get_actionvation(activation):
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'softmax':
        return tf.nn.softmax
    elif activation == 'sigmoid':
        return tf.nn.sigmoid
    elif activation is None:
        return None
    else:
        raise ValueError


def logic_regulation(x):
    return tf.reduce_sum(0.5 - tf.abs(0.5 - x))


def weight_regulation(x):
    return tf.nn.l2_loss(x)


def range_constraint(x):
    return tf.maximum(0.0, tf.minimum(x, 1.0))


def load_data(data_name):
    data_name = data_name.lower()
    if data_name == 'logicsyn':
        dataset = as_dataset(data_name, False)
        dataset.load_data()
        X_train = dataset.X_train
        y_train = dataset.y_train
        y_train = np.reshape(y_train, [-1])
    elif data_name == 'ml1m':
        dataset = as_dataset(data_name, True)
        dataset.load_data()
        X_train_raw = dataset.X_train
        y_train = dataset.y_train
        y_train = np.reshape(y_train, [-1])
        X_train_raw = np.delete(X_train_raw, 2, axis=1)
        X_train_raw[:, 2:] -= 3467
        X_train_raw[:, 2:] += 28
        X_train = np.zeros(shape=[X_train_raw.shape[0], np.max(X_train_raw)+1], dtype=np.int32)
        for index in range(X_train.shape[0]):
            X_train[index, X_train_raw[index]] = 1
    else:
        raise ValueError
    return X_train, y_train


def get_data(data_name, use_ratio, train_ratio):
    X_train, y_train = load_data(data_name)
    tot_samples = X_train.shape[0]
    num_small_train = int(tot_samples * use_ratio * train_ratio)
    num_small_valid = int(tot_samples * use_ratio * (1.0 - train_ratio))
    X_small_train = X_train[:num_small_train]
    y_small_train = y_train[:num_small_train]
    X_small_valid = X_train[num_small_train:num_small_train + num_small_valid]
    y_small_valid = y_train[num_small_train:num_small_train + num_small_valid]
    return X_small_train, y_small_train, X_small_valid, y_small_valid


def print_line(round, cur_sample, tot_samples, start_time, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc):
    progress = "".join([('#' if i *  tot_samples < cur_sample * 30 else '=') for i in range(30)])
    status = "\r\tRound:{} {}/{} [{}] Elapsed: {:.3f} seconds, train_loss:{:.3f} train_acc:{:.3f} train_auc:{:.3f} valid_loss:{:.3f} valid_acc:{:.3f} valid_auc:{:.3f}".format(
        round, cur_sample, tot_samples, progress, time.time() - start_time, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc
    )
    sys.stdout.write(status)
    if cur_sample == tot_samples:
        sys.stdout.write("\n")


def auc_score(y_true, y_score):
    if np.alltrue(y_true == 1) or np.alltrue(y_true == 0):
        return 1.0
    else:
        return roc_auc_score(y_true, y_score)


class Model:
    def __init__(self, input_dim, layers, lambda_negate=0.0, lambda_selector=0.0, lambda_propositions=0.0, lambda_weights=0.0):
        #  tensers_sets contains the set of some kind of tensors
        self.tensers_sets = dict()
        self.tensers_sets['propositions'] = []

        #  define forwarding phase
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, input_dim], name="inputs")  # batch * input_dim
        cur_layer = tf.cast(self.inputs, dtype=tf.float32)
        for i in range(len(layers)):
            layer_type, layer_params = layers[i]
            with tf.variable_scope('layer_{}_{}'.format(i, layer_type)):
                if layer_type == 'logic':
                    cur_layer = self.logic_layer(cur_layer, **layer_params)
                elif layer_type == 'dense':
                    cur_layer = self.dense_layer(cur_layer, **layer_params)
                else:
                    raise ValueError
        self.pred_scores = tf.reshape(cur_layer, [-1])  #  batch

        #  define loss
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
        negators_regu_list = [logic_regulation(v) for v in tf.get_collection('NEGATORS')]
        selectors_regu_list = [logic_regulation(v) for v in tf.get_collection('SELECTORS')]
        propositions_regu_list = [logic_regulation(v) for v in self.tensers_sets['propositions']]
        weights_regu_list = [weight_regulation(v) for v in tf.get_collection('WEIGHTS')]
        self.negators_regu = lambda_negate * tf.add_n(negators_regu_list) if len(negators_regu_list) != 0 else 0.0
        self.selectors_regu = lambda_selector * tf.add_n(selectors_regu_list) if len(selectors_regu_list) != 0 else 0.0
        self.propositions_regu = lambda_propositions * tf.add_n(propositions_regu_list) if len(propositions_regu_list) != 0 else 0.0
        self.weights_regu = lambda_weights * tf.add_n(weights_regu_list) if len(weights_regu_list) != 0 else 0.0
        self.regu_loss = self.negators_regu + self.selectors_regu + self.propositions_regu + self.weights_regu

        self.loss = tf.losses.log_loss(labels=self.labels, predictions=self.pred_scores) + self.regu_loss

    def logic_layer(self, input, units):  # input:  batch * input_dim
        input_dim = input.shape.as_list()[-1]
        negator = tf.get_variable('negator', shape=[input_dim, units], dtype=tf.float32,
                                  initializer=tf.initializers.random_normal(mean=0.5, stddev=0.5),
                                  trainable=True, collections=['NEGATORS', tf.GraphKeys.GLOBAL_VARIABLES],
                                  constraint=range_constraint)
        selector = tf.get_variable('selector', shape=[input_dim, units], dtype=tf.float32,
                                   initializer=tf.initializers.random_normal(mean=0.5, stddev=0.5),
                                   trainable=True, collections=['SELECTORS', tf.GraphKeys.GLOBAL_VARIABLES],
                                   constraint=range_constraint)
        input = tf.cast(tf.reshape(input, [-1, input_dim, 1]), dtype=tf.float32)
        negator = tf.reshape(negator, [1, input_dim, units])
        selector = tf.reshape(selector, [1, input_dim, units])
        after_negate = negator * (1.0 - input) + (1.0 - negator) * input
        after_select = after_negate * selector
        after_disjunct = tf.reduce_max(after_select, axis=1)
        self.tensers_sets['propositions'].append(after_disjunct)
        return after_disjunct

    def dense_layer(self, input, units, activation='relu'):
        input_dim = input.shape.as_list()[-1]
        weights = tf.get_variable('weight', shape=[input_dim, units], dtype=tf.float32,
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True, collections=['WEIGHTS', tf.GraphKeys.GLOBAL_VARIABLES])
        bias = tf.get_variable('bias', shape=[1, units], dtype=tf.float32,
                               initializer=tf.initializers.constant(0.0), trainable=True, collections=['BIAS', tf.GraphKeys.GLOBAL_VARIABLES])
        act = get_actionvation(activation)
        input = tf.reshape(input, shape=[-1, input_dim])
        if act is not None:
            output = act(tf.matmul(input, weights) + bias)
        else:
            output = tf.matmul(input, weights) + bias
        return output


def get_trainable_variable_number():
    total_number_parameters = 0
    for var in tf.trainable_variables():
        number_parameters = 1
        for dim in var.get_shape():
            number_parameters *= dim.value
        total_number_parameters += number_parameters
    return total_number_parameters


def train(layers, batch_size=32, eval_per_steps=30, max_rounds=50, use_ratio=0.4, train_ratio=0.8, lr=0.001, tensorboard_dir='statistics'):
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    #  load dataset
    X_train, y_train, X_valid, y_valid = get_data('ml1m', use_ratio=use_ratio, train_ratio=train_ratio)

    #  build graph
    graph = tf.Graph()
    sess = tf.Session(graph=graph, config=sess_config)
    writer = tf.summary.FileWriter(tensorboard_dir, graph=graph)
    with graph.as_default():
        learning_rate = tf.get_variable('lr', dtype=tf.float32, initializer=lr, trainable=False)
        model = Model(X_train.shape[1], layers)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = opt.minimize(model.loss)
        for var in tf.get_collection('NEGATORS'):
            tf.summary.histogram(var.name, var)
        for var in tf.get_collection('SELECTORS'):
            tf.summary.histogram(var.name, var)
        for var in tf.get_collection('WEIGHTS'):
            tf.summary.histogram(var.name, var)
        write_op = tf.summary.merge_all()

        print("Number of trainable parameters: {}\n".format(get_trainable_variable_number()))


    #  train information
    num_rounds = max_rounds
    tot_samples = X_train.shape[0]
    tot_steps = (tot_samples + batch_size - 1) // batch_size
    eval_per_steps = min(eval_per_steps, tot_steps)
    history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'valid_loss': [], 'valid_acc': [], 'valid_auc': []}


    with graph.as_default():
        sess.run(tf.global_variables_initializer())
        for round in range(num_rounds):
            start_time = time.time()
            train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            summary = sess.run(fetches=write_op)
            writer.add_summary(summary, round)
            for step in range(tot_steps):
                inputs = X_train[step * batch_size: (step + 1) * batch_size]
                labels = y_train[step * batch_size: (step + 1) * batch_size]
                _, pred_scores, loss = sess.run((train_op, model.pred_scores, model.loss), feed_dict={model.inputs: inputs, model.labels: labels})
                preds = np.where(pred_scores >= 0.5, np.ones_like(pred_scores, dtype=np.int32), np.zeros_like(pred_scores, dtype=np.int32))
                train_loss = (train_loss * step + np.mean(loss)) / (step + 1)
                train_auc = (train_auc * step + auc_score(labels, pred_scores)) / (step + 1)
                train_acc = (train_acc * step + np.count_nonzero(labels == preds) / batch_size) / (step + 1)

                if (step + 1) % eval_per_steps == 0 or step + 1 == tot_steps or step == 0:
                    val_pred_scores_list = []
                    val_loss_list = []
                    for i in range(X_valid.shape[0] // batch_size + 1):
                        batch_X = X_valid[i * batch_size:(i+1)*batch_size]
                        batch_y = y_valid[i * batch_size:(i+1)*batch_size]
                        pred, loss = sess.run((model.pred_scores, model.loss), feed_dict={model.inputs: batch_X, model.labels: batch_y})
                        val_pred_scores_list.append(pred)
                        val_loss_list.append(loss)
                    val_pred_scores = np.concatenate(val_pred_scores_list, axis=0)
                    val_loss = np.vstack(val_loss_list)
                    val_labels = y_valid
                    val_preds = np.where(val_pred_scores >= 0.5, np.ones_like(val_pred_scores, dtype=np.int32), np.zeros_like(val_pred_scores, dtype=np.int32))
                    valid_loss = np.mean(val_loss)
                    valid_acc = np.count_nonzero(val_labels == val_preds) / X_valid.shape[0]
                    valid_auc = auc_score(val_labels, val_pred_scores)
                print_line(round, min((step + 1) * batch_size, tot_samples), tot_samples, start_time, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_auc'].append(train_auc)
            history['valid_loss'].append(valid_loss)
            history['valid_acc'].append(valid_acc)
            history['valid_auc'].append(valid_auc)

        negators, selectors = None, None
        for var in tf.get_collection('NEGATORS'):
            if var.name.find('layer_0') != -1:
                negators = np.greater(sess.run(fetches=var), 0.5)
        for var in tf.get_collection('SELECTORS'):
            if var.name.find('layer_0') != -1:
                selectors = np.greater(sess.run(fetches=var), 0.5)
        for i in range(selectors.shape[1]):
            for j in range(selectors.shape[0]):
                if selectors[j][i]:
                    print('%s%i' % ('-' if negators[j][i] else '', j + 1), end=' ')
            print(' ')

    return history

def plot_histories(histories, run_name):
    # Plot training & validation accuracy values
    legends = []
    plt.title(run_name.split('/')[-1])
    filename = run_name + '.png'
    for curve_name, points in histories.items():
        plt.plot(points)
        legends.append(curve_name)
    plt.ylabel('Auc')
    plt.xlabel('Epoch')
    plt.legend(legends, loc='center right')
    plt.savefig(filename, format='png')
    plt.show()

#  construct layers
width = 6
layers = []
layers.append(('logic', {'units': width}))
layers.append(('logic', {'units': width}))
# layers.append(('logic', {'units': 1}))
layers.append(('dense', {'units': width, 'activation': 'relu'}))
layers.append(('dense', {'units': 1, 'activation': 'sigmoid'}))


history = train(layers=layers,
                batch_size=args.batch_size,
                eval_per_steps=args.eval_per_steps,
                max_rounds=args.max_rounds,
                use_ratio=args.use_ratio,
                train_ratio=args.train_ratio,
                lr=args.lr,
                tensorboard_dir=args.run_name)
with open(args.run_name + '.json', 'w') as f:
    json.dump(history, f, cls=MyEncoder)
plot_histories(history,  args.run_name)