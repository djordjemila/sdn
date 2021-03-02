from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import sys


def error_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print(*args, file=sys.stdout, **kwargs)


def terminate_on_nan(loss):
    if torch.isnan(loss).any():
        error_print("Terminating program -- NaN detected.")
        exit()


def count_pars(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list2string(list_to_parse):
    output = ""
    for list_elem in list_to_parse:
        output += str(list_elem) + "_"
    return output


class EMA(nn.Module):
    """ Exponential Moving Average.
    Note that we store shadow params as learnable parameters. We force torch.save() to store them properly..
    """

    def __init__(self, model, decay):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('EMA decay must in [0,1]')
        self.decay = decay
        self.shadow_params = nn.ParameterDict({})
        self.train_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[self._dotless(name)] = nn.Parameter(param.data.clone(), requires_grad=False)

    @staticmethod
    def _dotless(name):
        return name.replace('.', '^')

    @torch.no_grad()
    def update(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[self._dotless(name)].data = \
                        self.decay * self.shadow_params[self._dotless(name)].data + (1.0 - self.decay) * param.data

    @torch.no_grad()
    def assign(self, model):
        # ema assignment
        train_params_has_items = bool(self.train_params)
        if self.decay > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if train_params_has_items:
                        self.train_params[name].data.copy_(param.data)
                    else:
                        self.train_params[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[self._dotless(name)].data)

    @torch.no_grad()
    def restore(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.train_params[name].data)


class Crop2d(nn.Module):

    def __init__(self, num):
        super().__init__()
        self.num = num

    def forward(self, input):
        if self.num == 0:
            return input
        else:
            return input[:, :, self.num:-self.num, self.num:-self.num]


def weights_init(module):
    """ Weight initialization for different neural network components. """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.orthogonal_(module.weight)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        module.weight.data.normal_(0.0, 0.02)


def WN(module, norm=True):
    classname = module.__class__.__name__
    if norm:
        if classname.find('ConvTranspose') != -1:
            return weight_norm(module, dim=1, name='weight')
        elif classname.find('Conv') != -1:
            return weight_norm(module, dim=0, name='weight')
        else:
            return module
    else:
        return module


class MaskedConv2d(nn.Conv2d):
    """ Masked version of a regular 2D CNN. """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ARCNN(nn.Module):
    def __init__(self, num_layers, num_outputs, z_size, h_size):
        super().__init__()
        self.num_outputs = num_outputs
        self.conv_a = MaskedConv2d(mask_type='A', in_channels=z_size, out_channels=h_size,
                                   kernel_size=3, stride=1, padding=1)
        self.conv_b = []
        for i in range(num_layers):
            self.conv_b.append(nn.ELU(True))
            self.conv_b.append(MaskedConv2d(mask_type='B',
                                            in_channels=h_size,
                                            out_channels=z_size * num_outputs if i == (num_layers-1) else h_size,
                                            kernel_size=3, stride=1, padding=1))
        self.conv_b = nn.Sequential(*self.conv_b)

    def forward(self, x, context):
        x = self.conv_b(self.conv_a(x) + context)
        return list(x.chunk(self.num_outputs, 1))


class Quantize(object):
    """Quantize tensor images which are expected to be in [0, 1]. """

    def __init__(self, nbits=8):
        self.nbits = nbits

    def __call__(self, tensor):
        if self.nbits < 8:
            tensor = torch.floor(tensor * 255 / 2 ** (8 - self.nbits))
            tensor /= (2 ** self.nbits - 1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(nbits={0})'.format(self.nbits)


def run_cuda_diagnostics(requested_num_gpus):
    print("\nCUDA diagnostics:")
    print("-----------------")
    print("CUDA available? ", torch.cuda.is_available())
    print("Requested num devices: ", requested_num_gpus)
    print("Available num of devices: ", torch.cuda.device_count())
    print("CUDNN backend: ", torch.backends.cudnn.enabled)
    assert requested_num_gpus <= torch.cuda.device_count(), "Not enough GPUs available."


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class Reshape4x4(nn.Module):
    def forward(self, x):
        x = x.view(list(x.shape[:-1]) + [-1, 4, 4])
        return x


class Contiguous(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        return x


class EncWrapper(nn.Module):
    # Implemented for compatibility with disentanglement_lib evaluation
    def __init__(self, encoder_list):
        super().__init__()
        self.enc_list = nn.Sequential(*encoder_list)

    def forward(self, input):
        self.z_params = self.enc_list(input)
        mu, logvar = self.z_params.chunk(2, dim=1)
        return mu, logvar