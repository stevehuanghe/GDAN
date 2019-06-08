import os
import time
import pprint
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models.gdan import CVAE, Discriminator, Regressor
from utils.data_factory import DataManager
from utils.utils import load_data, update_values, get_datetime_str
from utils.logger import Logger, log_args


parser = argparse.ArgumentParser(description='argument parser')

parser.add_argument('-cfg', '--config', metavar='YAML', default=None,
                    help='path to yaml config file')

# files and directories
parser.add_argument('-dn', '--data_name', metavar='NAME', default='CUB',
                    choices=['CUB', 'SUN', 'APY', 'AWA1', 'AWA2', 'ImageNet'],
                    help='name of dataset')
parser.add_argument('-d', '--data_root', metavar='DIR', default='./ZSL-GBU/xlsa17/data',
                    help='path to data directory')
parser.add_argument('-r', '--result', metavar='DIR', default='./result',
                    help='path to result directory')
parser.add_argument('-f', '--logfile', metavar='DIR', default=None,
                    help='path to result directory')
parser.add_argument('-ckpt', '--ckpt_dir', metavar='STR', default='./checkpoints/',
                    help='checkpoint file')

parser.add_argument('-clf', '--classifier', metavar='STR', default='KNN', choices=['knn', 'svc'],
                    help='method for classification')

# hyper-parameters
parser.add_argument('-ns', '--num_samples', type=int, metavar='INT', default=500,
                    help='number of samples drawn for each unseen class')
parser.add_argument('-k', '--K', metavar='INT', type=int, default=1,
                    help='number of neighbors in kNN')
parser.add_argument('-c', '--C', metavar='FLOAT', type=float, default=1.0,
                    help='penalty for SVC')

# environment
parser.add_argument('-g', '--gpu', metavar='IDs', default='0',
                    help='what GPUs to use')

args = parser.parse_args()

# if yaml config exists, load and override default ones
if args.config is not None:
    with open(args.config, 'r') as fin:
        options_yaml = yaml.load(fin)
    update_values(options_yaml, vars(args))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

ts = get_datetime_str()
if args.logfile is None:
    args.logfile = 'log_valtest_' + args.data_name + '_' + ts + '.txt'


data_dir = Path(args.data_root)
cub_dir = data_dir / Path(args.data_name)
att_path = cub_dir / Path('att_splits.mat')
res_path = cub_dir / Path('res101.mat')

pprint.pprint(vars(args))

result_dir = Path(args.result)
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)

val_acc_file = str(result_dir / Path('val_acc_' + args.data_name + '_' + ts + '.txt'))
logfile = result_dir / Path(args.logfile)
logMaster = Logger(str(logfile))
log_args(str(logfile), args)


def main():
    val_acc = []
    best_H = 0
    best_model = None

    logger = logMaster.get_logger('main')

    logger.info('loading data...')
    att_feats, train_data, val_data, test_data, test_s_data, classes = load_data(att_path=att_path, res_path=res_path)

    ckpt_dir = Path(args.ckpt_dir)
    filenames = ckpt_dir.glob('gdan_*.pkl')

    def cmp_func(s):
        s = str(s).split('.')[0]
        num = int(s.split('_')[-1])
        return num
    if len(test_s_data) == 0:
        std = True
    else:
        std = False

    filenames = sorted(filenames, key=cmp_func)
    for checkpoint in filenames:

        macc = eval_model_val(checkpoint, logger, att_feats, train_data, val_data, classes)

        val_acc.append(macc)

        H = macc

        if H > best_H:
            best_H = H
            best_model = checkpoint

        logger.info(f'best: {best_model}, macc_u : {best_H:4.5}\n')

    macc_u, macc_s, H = eval_model_test(best_model, logger, att_feats, train_data, test_data, test_s_data, classes)
    logger.info(f'\ntest: {best_model}, gzsl unseen: {macc_u:4.5}, '
                f'gzsl seen: {macc_s:4.5}, gzsl H: {H:4.5}\n')

    with open(val_acc_file, 'w') as fout:
        for acc in val_acc:
            fout.write(str(acc) + '\n')

    logger.info('evaluation finished.')


def eval_model_val(checkpoint, logger, att_feats, train_data, val_data, classes):
    logger.info('building model...')

    states = torch.load(checkpoint)

    net = CVAE(x_dim=states['x_dim'], s_dim=states['s_dim'], z_dim=states['z_dim'], enc_layers=states['enc_layers'],
               dec_layers=states['dec_layers'])
    dis = Discriminator(x_dim=states['x_dim'], s_dim=states['s_dim'], layers=states['dis_layers'])
    reg = Regressor(x_dim=states['x_dim'], s_dim=states['s_dim'], layers=states['reg_layers'])

    net.cuda()
    dis.cuda()
    reg.cuda()

    logger.info(f'loading model from checkpoint: {checkpoint}')

    net.load_state_dict(states['gen'])
    dis.load_state_dict(states['dis'])
    reg.load_state_dict(states['reg'])

    logger.info('generating synthetic samples...')
    net.eval()
    samples = generate_samples(net, args.num_samples, att_feats[classes['val']], classes['val'])

    new_train_data = train_data + samples
    X, Y = zip(*new_train_data)
    X = np.array(X)
    Y = np.array(Y)

    if args.classifier == 'svc':
        clf = LinearSVC(C=args.C)
        logger.info('training linear SVC...')
    else:
        clf = KNeighborsClassifier(n_neighbors=args.K)
        logger.info('training kNN classifier')

    clf.fit(X=X, y=Y)

    test_X, test_Y = zip(*val_data)

    logger.info('predicting...')

    pred_Y = clf.predict(test_X)
    macc_u = cal_macc(truth=test_Y, pred=pred_Y)

    logger.info(f'gzsl macc_u: {macc_u:4.5}')
    return macc_u


def eval_model_test(checkpoint, logger, att_feats, train_data, test_data, test_s_data, classes):
    logger.info('building model...')

    states = torch.load(checkpoint)

    net = CVAE(x_dim=states['x_dim'], s_dim=states['s_dim'], z_dim=states['z_dim'], enc_layers=states['enc_layers'],
               dec_layers=states['dec_layers'])
    dis = Discriminator(x_dim=states['x_dim'], s_dim=states['s_dim'], layers=states['dis_layers'])
    reg = Regressor(x_dim=states['x_dim'], s_dim=states['s_dim'], layers=states['reg_layers'])

    net.cuda()
    dis.cuda()
    reg.cuda()

    logger.info(f'loading model from checkpoint: {checkpoint}')

    net.load_state_dict(states['gen'])
    dis.load_state_dict(states['dis'])
    reg.load_state_dict(states['reg'])

    logger.info('generating synthetic samples...')
    net.eval()
    samples = generate_samples(net, args.num_samples, att_feats[classes['test']], classes['test'])

    new_train_data = train_data + samples
    X, Y = zip(*new_train_data)
    X = np.array(X)
    Y = np.array(Y)

    if args.classifier == 'svc':
        clf = LinearSVC(C=args.C)
        logger.info('training linear SVC...')
    else:
        clf = KNeighborsClassifier(n_neighbors=args.K)
        logger.info('training kNN classifier')

    clf.fit(X=X, y=Y)

    test_X, test_Y = zip(*test_data)

    logger.info('predicting...')

    pred_Y = clf.predict(test_X)
    macc_u = cal_macc(truth=test_Y, pred=pred_Y)

    if len(test_s_data) > 0:
        test_Xs, test_Ys = zip(*test_s_data)
        pred_Ys = clf.predict(test_Xs)
        macc_s = cal_macc(truth=test_Ys, pred=pred_Ys)
    else:
        macc_s = 0.0

    if macc_s + macc_u == 0.0:
        H = 0.0
    else: 
        H = 2 * macc_s * macc_u / (macc_s + macc_u)

    logger.info(f'gzsl unseen: {macc_u:4.5}, gzsl seen: {macc_s:4.5}, gzsl H: {H:4.5}\n')
    return macc_u, macc_s, H


def generate_samples(net, num_samples, class_emb, labels):
    class_emb = list(class_emb)
    data = []
    for i in range(len(class_emb)):
        for _ in range(num_samples):
            feats = Variable(torch.from_numpy(class_emb[i].reshape(1, -1)).float()).cuda()
            sample = net.sample(feats).cpu().data.numpy().reshape(-1)
            data.append((sample, labels[i]))
    return data


def cal_macc(*, truth, pred):
    assert len(truth) == len(pred)
    count = {}
    total = {}
    labels = list(set(truth))
    for label in labels:
        count[label] = 0
        total[label] = 0

    for y in truth:
        total[y] += 1

    correct = np.nonzero(np.asarray(truth) == np.asarray(pred))[0]

    for c in correct:
        idx = truth[c]
        count[idx] += 1

    macc = 0
    num_class = len(labels)
    for key in count.keys():
        if total[key] == 0:
            num_class -= 1
        else:
            macc += count[key] / total[key]
    macc /= num_class
    return macc


if __name__ == '__main__':
    main()
