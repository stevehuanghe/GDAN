import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, x_dim=2048, s_dim=312, z_dim=100, enc_layers='1200 600', dec_layers='600'):
        super(CVAE2, self).__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.z_dim = z_dim

        enc_layers = enc_layers.split()
        encoder = []
        for i in range(len(enc_layers)):
            num_hidden = int(enc_layers[i])
            pre_hidden = int(enc_layers[i-1])
            if i == 0:
                encoder.append(nn.Linear(x_dim, num_hidden))
                encoder.append(nn.ReLU())
            else:
                encoder.append(nn.Dropout(p=0.3))
                encoder.append(nn.Linear(pre_hidden, num_hidden))
                encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        last_hidden = int(enc_layers[-1])
        self.mu_net = nn.Sequential(
            nn.Linear(last_hidden, z_dim)
        )

        self.sig_net = nn.Sequential(
            nn.Linear(last_hidden, z_dim)
        )

        dec_layers = dec_layers.split()
        decoder = []
        for i in range(len(dec_layers)):
            num_hidden = int(dec_layers[i])
            pre_hidden = int(dec_layers[i-1])
            if i == 0:
                decoder.append(nn.Linear(z_dim + s_dim, num_hidden))
                decoder.append(nn.ReLU())
            else:
                decoder.append(nn.Linear(pre_hidden, num_hidden))
                decoder.append(nn.ReLU())
            if i == len(dec_layers) - 1:
                decoder.append(nn.Linear(num_hidden, x_dim))
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)


    def encode(self, X):
        hidden = self.encoder(X)
        mu = self.mu_net(hidden)
        log_sigma = self.sig_net(hidden)
        return mu, log_sigma

    def decode(self, S, mu, log_sigma):
        eps = torch.rand(mu.size()).cuda()
        Z = mu + torch.exp(log_sigma / 2) * eps
        ZS = torch.cat([S, Z], dim=1)
        Xp = self.decoder(ZS)
        return Xp

    def forward(self, X, S):
        hidden = self.encoder(X)
        mu = self.mu_net(hidden)
        log_sigma = self.sig_net(hidden)
        eps = torch.rand(mu.size())
        eps = eps.cuda()
        Z = mu + torch.exp(log_sigma / 2) * eps
        ZS = torch.cat([S, Z], dim=1)
        Xp = self.decoder(ZS)
        return Xp, mu, log_sigma

    def sample(self, S):
        Z = torch.rand([S.size()[0], self.z_dim]).cuda()
        ZS = torch.cat([S, Z], dim=1)
        return self.decoder(ZS)

    def vae_loss(self, X, Xp, mu, log_sigma):
        reconstruct_loss = 0.5 * torch.sum(torch.pow(X-Xp, 2), 1)
        KL_divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1 - log_sigma, 1)
        return torch.mean(reconstruct_loss + KL_divergence)


class Generator(nn.Module):
    def __init__(self, x_dim=2048, s_dim=312, z_dim=100):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        total_dim = s_dim + z_dim
        self.FCN = nn.Sequential(
            nn.Linear(total_dim, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200, x_dim),
            nn.ReLU()
        )
        self.FCN.apply(init_weights)

    def forward(self, S):
        Z = torch.rand([S.size()[0], self.z_dim]).cuda()
        ZS = torch.cat([S, Z], dim=1)
        return self.FCN(ZS)


class Discriminator(nn.Module):
    def __init__(self, x_dim=2048, s_dim=312, layers='1200 600'):
        super(Discriminator, self).__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        total_dim = x_dim + s_dim
        layers = layers.split()
        fcn_layers = []

        for i in range(len(layers)):
            pre_hidden = int(layers[i-1])
            num_hidden = int(layers[i])
            if i == 0:
                fcn_layers.append(nn.Linear(total_dim, num_hidden))
                fcn_layers.append(nn.ReLU())
            else:
                fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                fcn_layers.append(nn.ReLU())

            if i == len(layers) - 1:
                fcn_layers.append(nn.Linear(num_hidden, 1))

        self.FCN = nn.Sequential(*fcn_layers)
        self.mse_loss = nn.MSELoss()

    def forward(self, X, S):
        XS = torch.cat([X, S], 1)
        return self.FCN(XS)

    def dis_loss(self, *, X, Xp, S, Sp):
        true_scores = self.forward(X, S)
        fake_scores = self.forward(Xp, S)
        ctrl_socres = self.forward(X, Sp)
        return self.mse_loss(true_scores, 1) + self.mse_loss(fake_scores, 0) + self.mse_loss(ctrl_socres, 0)


class Regressor(nn.Module):
    def __init__(self, x_dim=2048, s_dim=312, layers='512'):
        super(Regressor, self).__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        layers = layers.split()
        fcn_layers = []

        for i in range(len(layers)):
            pre_hidden = int(layers[i - 1])
            num_hidden = int(layers[i])
            if i == 0:
                fcn_layers.append(nn.Linear(x_dim, num_hidden))
                fcn_layers.append(nn.ReLU())
            else:
                fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                fcn_layers.append(nn.ReLU())

            if i == len(layers) - 1:
                fcn_layers.append(nn.Linear(num_hidden, s_dim))
                fcn_layers.append(nn.ReLU())

        self.FCN = nn.Sequential(*fcn_layers)
        self.criterion = nn.MSELoss()

    def forward(self, X):
        return self.FCN(X)

    def reg_loss(self, Sp, S, Xp, Xpp):
        return self.criterion(Sp, S) + self.criterion(Xp, Xpp)


