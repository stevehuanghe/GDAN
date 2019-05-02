import argparse

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
parser.add_argument('-cd', '--ckpt_dir', metavar='DIR', default='./checkpoints',
                    help='path to checkpoint directory')
parser.add_argument('-v', '--vae_ckpt', metavar='FILE', default='./checkpoints/cvae_800.pkl',
                    help='path to data directory')

# flags
parser.add_argument('--debug', action='store_true',
                    help='set this flag to enter debug mode')


# environment
parser.add_argument('-g', '--gpu', metavar='IDs', default='1',
                    help='which GPUs to use')
parser.add_argument('--save_epoch', default=10, type=int, metavar='INT',
                    help='save every N epochs')

# optimizer related
parser.add_argument('-e', '--epoch', default=1000, type=int, metavar='INT',
                    help='number of total epochs to run')
parser.add_argument('-st', '--steps', default=-1, type=int, metavar='INT',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch', default=128, type=int, metavar='INT',
                    help='number of batch size')
parser.add_argument('-lr', '--learning_rate', default=0.00001, type=float, metavar='FLOAT',
                    help='learning rate')
parser.add_argument('-dt', '--d_iter', default=1, type=int, metavar='INT',
                    help='number of steps that the discriminator updates')
parser.add_argument('-gt', '--g_iter', default=1, type=int, metavar='INT',
                    help='number of steps that the generator updates')

# model related
parser.add_argument('--enc', default='1200 600', metavar='INT',
                    help='dimension of encoder')
parser.add_argument('--dec', default='600', metavar='INT',
                    help='dimension of decoder')
parser.add_argument('--reg', default='512', metavar='INT',
                    help='dimension of regressor')
parser.add_argument('--dis', default='1200 600', metavar='INT',
                    help='dimension of discriminator')

parser.add_argument('-dx', '--x_dim', default=2048, type=int, metavar='INT',
                    help='dimension of image feature')
parser.add_argument('-ds', '--s_dim', default=312, type=int, metavar='INT',
                    help='dimension of attribute feature')
parser.add_argument('-dz', '--z_dim', default=100, type=int, metavar='INT',
                    help='dimension of noise vector')

parser.add_argument('--gan_model', default='lsgan', choices=['lsgan', 'wgan-gp'],
                    help='training method of GAN, LSGAN or WGAN-GP')

parser.add_argument('-t1', '--theta1', default=0.1, type=float, metavar='FLOAT',
                    help='theta for cycle loss')
parser.add_argument('-t2', '--theta2', default=0.1, type=float, metavar='FLOAT',
                    help='theta for regression loss')
parser.add_argument('-t3', '--theta3', default=0.1, type=float, metavar='FLOAT',
                    help='theta for regressor adv loss')

