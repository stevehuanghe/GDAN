
import os
import time
import pprint
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models.gdan import CVAE, Discriminator, Regressor
from utils.config_gdan import parser
from utils.data_factory import DataManager
from utils.utils import load_data, update_values
from utils.logger import Logger


args = parser.parse_args()

# if yaml config exists, load and override default ones
if args.config is not None:
    with open(args.config, 'r') as fin:
        options_yaml = yaml.load(fin)
    update_values(options_yaml, vars(args))

data_dir = Path(args.data_root)
cub_dir = data_dir / Path(args.data_name)
att_path = cub_dir / Path('att_splits.mat')
res_path = cub_dir / Path('res101.mat')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

pprint.pprint(vars(args))

save_dir = Path(args.vae_dir)
if not save_dir.is_dir():
    save_dir.mkdir(parents=True)

log_file = save_dir / Path('log_cvae.txt')
logMaster = Logger(str(log_file))


def main():
    logger = logMaster.get_logger('main')
    logger.info('loading data...')
    att_feats, train_data, val_data, test_data, test_s_data, classes = load_data(att_path=att_path, res_path=res_path)

    logger.info('building model...')

    cvae = CVAE(x_dim=args.x_dim, s_dim=args.s_dim, z_dim=args.z_dim, enc_layers=args.enc, dec_layers=args.dec)  # , theta=args.theta

    cvae.cuda()

    cvae_opt = optim.Adam(cvae.parameters(), lr=args.learning_rate, weight_decay=0.01)  #

    train_manager = DataManager(train_data, args.epoch, args.batch)

    logger.info('start training...')
    for epoch in range(1000):
        running_loss = 0
        t1 = time.time()
        cvae.train()
        for batch in tqdm(range(train_manager.num_batch), leave=False, ncols=70, unit='b'):

            data = train_manager.get_batch()
            X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
            Y = [item[1] for item in data]
            S = Variable(torch.from_numpy(att_feats[Y])).float().cuda()

            Xp, mu, log_sigma = cvae.forward(X, S)

            loss_vae = cvae.vae_loss(X, Xp, mu, log_sigma)

            loss_vae.backward()
            cvae_opt.step()

            running_loss += loss_vae.cpu().data.numpy()
        running_loss /= train_manager.num_batch
        elapsed = (time.time() - t1)/60.0

        if (epoch+1) % 10 == 0:
            filename = 'cvae_' + str(epoch+1) + '.pkl'
            save_path = save_dir / Path(filename)
            states = {}
            states['model'] = cvae.state_dict()
            states['z_dim'] = args.z_dim
            states['x_dim'] = args.x_dim
            states['s_dim'] = args.s_dim
            states['optim'] = cvae.state_dict()
            torch.save(states, str(save_path))
            logger.info(f'epoch: {epoch+1:4}, loss: {running_loss: .5}')
    logger.info('program finished')


if __name__ == '__main__':
    main()
