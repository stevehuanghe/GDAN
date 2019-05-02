
import os
import time
import yaml
import pprint
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable

from models.gdan import CVAE, Generator, Discriminator, Regressor
from utils.config_gdan import parser
from utils.data_factory import DataManager
from utils.utils import load_data, update_values, get_negative_samples
from utils.logger import Logger, log_args


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

save_dir = Path(args.ckpt_dir)
if not save_dir.is_dir():
    save_dir.mkdir(parents=True)

result_dir = Path(args.result)
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)


result_path = save_dir / Path('gdan_loss.txt')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

pprint.pprint(vars(args))

log_path = save_dir / Path('gdan_log.txt')
print('log file:', log_path)
logMaster = Logger(str(log_path))
log_args(log_path, args)


def main():
    logger = logMaster.get_logger('main')
    logger.info('loading data...')
    att_feats, train_data, val_data, test_data, test_s_data, classes = load_data(att_path=att_path, res_path=res_path)

    logger.info('building model...')

    gen = CVAE(x_dim=args.x_dim, s_dim=args.s_dim, z_dim=args.z_dim, enc_layers=args.enc, dec_layers=args.dec)
    gen.train()
    states = torch.load(args.vae_ckpt)
    gen.load_state_dict(states['model'])

    dis = Discriminator(x_dim=args.x_dim, s_dim=args.s_dim, layers=args.dis)
    reg = Regressor(x_dim=args.x_dim, s_dim=args.s_dim, layers=args.reg)

    gen.cuda()
    dis.cuda()
    reg.cuda()

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    adam_betas = (0.8, 0.999)
    gen_opt = optim.Adam(gen.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=adam_betas)
    dis_opt = optim.Adam(dis.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=adam_betas)
    reg_opt = optim.Adam(reg.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=adam_betas)

    train_manager = DataManager(train_data, args.epoch, args.batch, infinite=True)

    ones = Variable(torch.ones([args.batch, 1]), requires_grad=False).float().cuda()
    zeros = Variable(torch.zeros([args.batch, 1]), requires_grad=False).float().cuda()

    loss_history = []
    logger.info('start training...')
    for epoch in range(args.epoch):
        running_loss = 0
        t1 = time.time()
        d_total_loss = 0.0
        g_total_loss = 0.0
        cyc_total_loss = 0.0
        r_total_loss = 0.0
        rd_total_loss = 0.0
        vae_total_loss = 0.0
        g_scores = 0.0

        if args.steps == -1:
            steps = train_manager.num_batch
        else:
            steps = args.steps

        for batch in tqdm(range(steps), leave=False, ncols=70, unit='b'):
            for i in range(args.d_iter):
                dis.zero_grad()

                # get true data
                data = train_manager.get_batch()
                X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
                Y = [item[1] for item in data]
                S = Variable(torch.from_numpy(att_feats[Y])).float().cuda()
                Yc = get_negative_samples(Y, classes['train'])
                Sc = Variable(torch.from_numpy(att_feats[Yc])).float().cuda()

                # get fake data
                Xp, _, _ = gen.forward(X, S)
                Xp = Xp.detach()  # fix the generator
                Xpp = gen.sample(S).detach()
                Sp = reg.forward(X).detach()  # fix the regressor

                # get scores
                true_scores = dis.forward(X, S)
                fake_scores = dis.forward(Xp, S)
                fake_scores2 = dis.forward(Xpp, S)
                reg_scores = dis.forward(X, Sp)
                ctrl_scores = dis.forward(X, Sc)

                # calculate loss
                d_loss = mse_loss(true_scores, ones) + mse_loss(fake_scores, zeros) + args.theta3 * mse_loss(reg_scores, zeros) \
                         + mse_loss(ctrl_scores, zeros) 

                d_loss.backward()
                dis_opt.step()

                d_total_loss += d_loss.cpu().data.numpy()

            for i in range(args.g_iter):
                gen.zero_grad()
                reg.zero_grad()

                # get true data
                data = train_manager.get_batch()
                X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
                Y = [item[1] for item in data]
                S = Variable(torch.from_numpy(att_feats[Y])).float().cuda()

                # get fake data
                Xp, mu, log_sigma = gen.forward(X, S)
                Xp2 = gen.sample(S)
                Sp = reg.forward(X)
                Spp = reg.forward(Xp)
                Xpp, _, _ = gen.forward(X, Sp)

                # get scores
                fake_scores = dis.forward(Xp, S)
                fake_scores2 = dis.forward(Xp2, S)
                reg_scores = dis.forward(X, Sp)

                # calculate loss
                vae_loss = gen.vae_loss(X=X, Xp=Xp, mu=mu, log_sigma=log_sigma)
                cyc_loss = mse_loss(Spp, S) + mse_loss(Xpp, X)

                g_loss = mse_loss(fake_scores, ones) 
                r_loss = mse_loss(Sp, S)
                rd_loss = mse_loss(reg_scores, ones)

                total_loss = vae_loss + g_loss + args.theta1 * cyc_loss + args.theta2 * r_loss + args.theta3 * rd_loss
                total_loss.backward()

                gen_opt.step()
                reg_opt.step()

                vae_total_loss += vae_loss.cpu().data.numpy()
                g_total_loss += g_loss.cpu().data.numpy()
                cyc_total_loss += cyc_loss.cpu().data.numpy()
                r_total_loss += r_loss.cpu().data.numpy()
                rd_total_loss += rd_loss.cpu().data.numpy()
                g_scores += np.mean(fake_scores.cpu().data.numpy())

        g_total_steps = steps * args.g_iter
        d_total_steps = steps * args.d_iter
        vae_avg_loss = vae_total_loss / g_total_steps
        g_avg_loss = g_total_loss / g_total_steps
        cyc_avg_loss = cyc_total_loss / g_total_steps
        r_avg_loss = r_total_loss / g_total_steps
        rd_avg_loss = rd_total_loss / g_total_steps
        d_avg_loss = d_total_loss / d_total_steps
        g_avg_score = g_scores / g_total_steps
        loss_history.append(f'{g_avg_loss:.4}\t{d_avg_loss:.4}\t{cyc_avg_loss:.4}\t{r_avg_loss:.4}\t'
                            f'{rd_avg_loss:.4}\t{g_avg_score:.4}\t{vae_avg_loss:.4}\n')
        elapsed = (time.time() - t1)/60.0

        if (epoch+1) % 10 == 0 or epoch == 0:
            filename = 'gdan_' + str(epoch + 1) + '.pkl'
            save_path = save_dir / Path(filename)
            states = dict()
            states['epoch'] = epoch + 1
            states['gen'] = gen.state_dict()
            states['dis'] = dis.state_dict()
            states['reg'] = reg.state_dict()
            states['enc_layers'] = args.enc
            states['dec_layers'] = args.dec
            states['reg_layers'] = args.reg
            states['dis_layers'] = args.dis
            states['z_dim'] = args.z_dim
            states['x_dim'] = args.x_dim
            states['s_dim'] = args.s_dim
            states['gen_opt'] = gen_opt.state_dict()
            states['dis_opt'] = dis_opt.state_dict()
            states['reg_opt'] = reg_opt.state_dict()
            states['theta1'] = args.theta1
            states['theta2'] = args.theta2
            states['theta3'] = args.theta3

            torch.save(states, str(save_path))
            logger.info(f'epoch: {epoch+1:4}, g_loss: {g_avg_loss: .4}, d_loss: {d_avg_loss: .4}, \n'
                        f'cyc_loss: {cyc_avg_loss: .4}, r_loss: {r_avg_loss: .4}, rd_loss: {rd_avg_loss: .4}, '
                        f'g_score: {g_avg_score:.4}, vae loss: {vae_avg_loss:.4}')

    with result_path.open('w') as fout:
        for s in loss_history:
            fout.write(s)

    logger.info('program finished')




if __name__ == '__main__':
    main()
