import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell
import math


class SDNCell(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gru = GRUCell(input_size=3*num_features, hidden_size=num_features)

    def forward(self, neighboring_features, features):
        """ Update current features based on neighboring features """
        return self.gru(torch.cat(neighboring_features, dim=1), features)


class _SDNLayer(nn.Module):

    def __init__(self, num_features, dir=0):
        super().__init__()
        self.num_features = num_features
        self.cell = SDNCell(num_features)
        if dir == 0:
            self.forward = self.forward0
        elif dir == 1:
            self.forward = self.forward1
        elif dir == 2:
            self.forward = self.forward2
        else:
            self.forward = self.forward3
        self.zero_pad = None

    def _get_zero_pad(self, batch, device):
        if self.zero_pad is None or self.zero_pad.shape[0] != batch:
            self.zero_pad = torch.zeros((batch, self.num_features, 1), device=device)  # no grad ??
        return self.zero_pad

    def forward0(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(1, dim):
            neighboring_features = torch.cat([zero_pad, features[:, :, :, d - 1], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, :, d] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, :, d].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features

    def forward1(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(dim - 2, -1, -1):
            neighboring_features = torch.cat([zero_pad, features[:, :, :, d + 1], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, :, d] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, :, d].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features

    def forward2(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(1, dim):
            neighboring_features = torch.cat([zero_pad, features[:, :, d - 1, :], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, d, :] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, d, :].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features

    def forward3(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(dim - 2, -1, -1):
            neighboring_features = torch.cat([zero_pad, features[:, :, d + 1, :], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, d, :] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, d, :].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features


class SDN(nn.Module):
    def __init__(self, in_ch, out_ch, num_features, dirs, kernel_size, stride, padding, upsample):
        super().__init__()
        # project-in network
        cnn_module = nn.ConvTranspose2d if upsample else nn.Conv2d
        self.pre_cnn = cnn_module(in_ch, num_features, kernel_size, stride, padding)
        # update network
        sdn_update_blocks = []
        for dir in dirs:
            sdn_update_blocks.append(_SDNLayer(num_features, dir=dir))
        self.sdn_update_network = nn.Sequential(*sdn_update_blocks)
        # project-out network
        self.post_cnn = nn.Conv2d(num_features, out_ch, 1)

    def forward(self, x):
        # (I) project-in step
        x = self.pre_cnn(x)
        x = torch.tanh(x)
        # (II) update step
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.sdn_update_network(x)
        x = x.contiguous(memory_format=torch.contiguous_format)
        # (III) project-out step
        x = self.post_cnn(x)
        return x


class ResSDN(nn.Module):

    def __init__(self, in_ch, out_ch, num_features, dirs, kernel_size, stride, padding, upsample):
        super().__init__()
        self.sdn = SDN(in_ch, 2 * out_ch, num_features, dirs, kernel_size, stride, padding, upsample)
        cnn_module = nn.ConvTranspose2d if upsample else nn.Conv2d
        self.cnn = cnn_module(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, input):
        cnn_out = self.cnn(input)
        sdn_out, gate = self.sdn(input).chunk(2, 1)
        gate = torch.sigmoid(gate)
        return gate * cnn_out + (1-gate) * sdn_out


class LadderLayer(nn.Module):
    def __init__(self, post_model, prior_model, z_size, h_size, free_bits, downsample, sdn_num_features,
                 sdn_dirs_a, sdn_dirs_b, use_sdn, sampling_temperature):
        super().__init__()

        # initializations
        self.post_model = post_model
        self.prior_model = prior_model
        self.logqzx_params_up = None
        self.free_bits = free_bits
        self.downsample = downsample
        self.use_sdn = use_sdn
        self.sampling_temperature = sampling_temperature
        self.act = nn.ELU(True)

        # infer CNN parameters based on whether we do downsampling or not
        kernel_size, stride, padding = (4, 2, 1) if downsample else (3, 1, 1)

        # create modules for bottom-up pass
        self.up_a_layout = [h_size, z_size * post_model.params_per_dim()]
        self.up_a = nn.Conv2d(h_size, sum(self.up_a_layout), kernel_size, stride, padding)
        self.up_b = nn.Conv2d(h_size, 2*h_size, 3, 1, 1)

        # create modules for top-down pass
        self.down_a_layout = [h_size, z_size * post_model.params_per_dim(), z_size * prior_model.params_per_dim()]
        if use_sdn:
            self.down_a = ResSDN(in_ch=h_size, out_ch=sum(self.down_a_layout), num_features=sdn_num_features,
                                 dirs=sdn_dirs_a, kernel_size=3, stride=1, padding=1, upsample=False)
            self.down_b = ResSDN(in_ch=h_size + z_size, out_ch=2 * h_size, num_features=sdn_num_features,
                                 dirs=sdn_dirs_b, kernel_size=kernel_size, stride=stride, padding=padding,
                                 upsample=downsample)
        else:
            self.down_a = nn.Conv2d(h_size, sum(self.down_a_layout), 3, 1, 1)
            cnn_module = nn.ConvTranspose2d if downsample else nn.Conv2d
            self.down_b = cnn_module(h_size + z_size, 2 * h_size, kernel_size, stride, padding)

    def up(self, input):

        x = self.act(input)
        x = self.up_a(x)

        h, self.logqzx_params_up = x.split(self.up_a_layout, 1)

        h = self.act(h)
        h = self.up_b(h)

        h, gate = h.chunk(2, 1)
        gate = torch.sigmoid(gate-1)

        # possibly downsample input
        if self.downsample:
            input = F.interpolate(input, scale_factor=0.5)

        return (1-gate) * input + gate * h

    def down(self, input, sample=False, temperature=1.0, fixed_z=None):

        x = self.act(input)
        x = self.down_a(x)

        h_det, logqzx_params_down, logpz_params = x.split(self.down_a_layout, 1)

        if sample:
            z = self.prior_model.sample_once(logpz_params, temperature)
            if fixed_z is not None:
                z = z * temperature + fixed_z * (1-temperature)
            kl = kl_obj = torch.zeros(input.size(0), device=input.device)
        elif fixed_z is not None:
            z = fixed_z
            kl = kl_obj = torch.zeros(input.size(0), device=input.device)
        else:
            # merge posterior parameters
            q_params = self.logqzx_params_up + logqzx_params_down
            # sample and compute E[log q(z|x)]
            z, logqzx = self.post_model.reparameterize(q_params)
            # compute E[log p(z)]
            logpz = self.prior_model.conditional_log_prob(logpz_params, z)
            # compute KL[p(z|x)||p(z)]
            kl = kl_obj = logqzx - logpz
            # free bits are computed per layer
            kl_extra = (max(kl_obj.mean(), self.free_bits) - kl_obj.mean()) / kl_obj.size(0)
            kl_obj = kl_obj + kl_extra

        h = torch.cat((z, h_det), 1)
        h = self.act(h)
        h = self.down_b(h)

        h, gate = h.chunk(2, 1)
        gate = torch.sigmoid(gate-1)

        # possibly upsample input
        if self.downsample:
            input = F.interpolate(input, scale_factor=2.0)

        return (1-gate) * input + gate * h, kl, kl_obj, z
