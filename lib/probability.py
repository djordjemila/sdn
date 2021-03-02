from abc import abstractmethod, ABC
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from lib.utils import ARCNN, Quantize
from torchvision.transforms import Normalize, Compose, ToTensor


class ProbabilisticModel(nn.Module):

    def __init__(self, **kw):
        super().__init__()

    @abstractmethod
    def params_per_dim(self):
        """ how many parameters are computed per dimension. """

    @abstractmethod
    def get_mean(self, params):
        """ get mean value for specified parameters. """

    def get_mode(self, params):
        """ mean by default. """
        return self.get_mean(params)

    def get_most_probable_output(self, params):
        """ mode by default. """
        return self.get_mode(params)

    @abstractmethod
    def sample_once(self, params, sampling_temperature=1.0):
        """ sample from the model given parameters. """

    def sample(self, params, num_times):
        """ sample multiple times. """
        samples = []
        for i in range(num_times):
            samples.append(self.sample_once(params))
        return samples

    def reparameterize(self, params):
        """ compute distance from prior and return sample """
        return self.sample_once(params), torch.tensor([0.0], device=self.device)

    def log_prob(self, sample):
        """ distance from prior """
        return torch.tensor([0.0], device=self.device)

    @abstractmethod
    def conditional_log_prob(self, params, sample):
        """ compute conditional nll for a given sample. """


class LatentProbabilisticModel(ProbabilisticModel, ABC):

    def __init__(self, **kw):
        super().__init__()


class IsoGaussian(LatentProbabilisticModel):

    def __init__(self, **kw):
        super().__init__(**kw)

    def params_per_dim(self):
        """ we have mean and log-variance, so 2 parameters. """
        return 2

    def get_mean(self, params):
        """ Means are estimated parameters. """
        mean, _ = params.chunk(2, dim=1)
        return mean

    def sample_from_prior(self, shape, sampling_temperature=1.0):
        """ Sample from prior. """
        mean, lv = torch.zeros(shape), torch.zeros(shape)
        std = torch.exp(0.5 * lv) * sampling_temperature
        eps = torch.randn_like(std)
        return mean + eps * std

    def sample_once(self, params, sampling_temperature=1.0):
        """ Sample with reparameterization trick to enable differentiation. """
        mean, lv = params.chunk(2, dim=1)
        std = torch.exp(0.5 * lv) * sampling_temperature
        eps = torch.randn_like(std)
        return mean + eps * std

    def reparameterize(self, params, reduce=True):
        """ compute distance from prior and return sample """
        mean, lv = params.chunk(2, dim=1)
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        # compute sample
        sample = mean + eps * std
        # compute log probability
        log_prob = - torch.log(std) - 0.5 * np.log(2 * np.pi) - 0.5 * eps.pow(2)
        if reduce:
            return sample, log_prob.view(log_prob.size(0), -1).sum(dim=1)
        else:
            return sample, log_prob

    def log_prob(self, sample):
        log_prob = - 0.5 * np.log(2 * np.pi) - 0.5 * sample.pow(2)
        return log_prob.view(log_prob.size(0), -1).sum(dim=1)

    def conditional_log_prob(self, params, sample, reduce=True):
        mean, lv = params.chunk(2, dim=1)
        dist = D.Normal(mean, torch.exp(0.5 * lv))
        log_prob = dist.log_prob(sample)
        if reduce:
            return log_prob.view(log_prob.size(0), -1).sum(dim=1)
        else:
            return log_prob


class IAF(LatentProbabilisticModel):

    def __init__(self, z_size, context_size=100, num_flows=1, num_layers=2, **kw):
        super().__init__(**kw)
        self.z_size = z_size
        # make sure context is divisible by number of stochastic filters, for coding convenience
        self.context_size = max((context_size // self.z_size), 1) * self.z_size
        self.num_flows = num_flows
        self.arnn = ARCNN(num_layers=num_layers,  num_outputs=2, z_size=self.z_size, h_size=self.context_size)

    def unpack_params(self, params):
        """ parameter unpacking """
        mean, lv, context = params.split([self.z_size, self.z_size, self.context_size], dim=1)
        return mean, lv, context

    def flow(self, z, context):
        logsd = 0
        for i in range(self.num_flows):
            x = self.arnn(z, context)
            arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
            z = (z - arw_mean) / torch.exp(arw_logsd)
            logsd += arw_logsd
        return z, logsd

    def params_per_dim(self):
        """ we have mean and log-variance, and context. """
        return 2 + self.context_size // self.z_size

    def get_mean(self, params):
        mean, _, _ = self.unpack_params(params)
        return mean

    def sample_once(self, params, sampling_temperature=1.0):
        """ Sample with reparameterization trick to enable differentiation. """
        # unpack parameters
        mean, lv, context = self.unpack_params(params)
        # sample z from a Gaussian
        std = torch.exp(0.5 * lv) * sampling_temperature
        eps = torch.randn_like(std)
        sample = mean + eps * std
        # transform the sample using flow
        sample, _ = self.flow(sample, context)
        # return final sample
        return sample

    def reparameterize(self, params, reduce=True):
        """ Sample with reparameterization trick to enable differentiation. """
        # unpack parameters
        mean, lv, context = self.unpack_params(params)
        # sample z from a Gaussian
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        sample = mean + eps * std
        # compute log probability
        log_prob = - torch.log(std) - 0.5 * np.log(2 * np.pi) - 0.5 * eps.pow(2)
        # transform the sample using flow
        sample, logsd = self.flow(sample, context)
        # adjust log probability
        log_prob = log_prob + logsd
        # return final sample and log probability
        if reduce:
            return sample, log_prob.view(log_prob.size(0), -1).sum(dim=1)
        else:
            return sample, log_prob

    def conditional_log_prob(self, params, sample, reduce=True):
        """ not needed in our case, for the moment. """
        # unpack parameters
        mean, lv, context = self.unpack_params(params)
        # compute log_prob under gaussian
        dist = D.Normal(mean, torch.exp(0.5 * lv))
        log_prob = dist.log_prob(sample)
        # perform normalizing flow
        _, logsd = self.flow(sample, context)
        # adjust log probability
        log_prob = log_prob + logsd
        # return final sample and log probability
        if reduce:
            return log_prob.view(log_prob.size(0), -1).sum(dim=1)
        else:
            return log_prob


class ObservationProbabilisticModel(ProbabilisticModel, ABC):

    def __init__(self, channels, image_size, nbits=8, **kw):
        super().__init__(**kw)
        self.channels = channels
        self.image_size = image_size
        self.nbits = nbits

    def get_transform(self):
        """ Empty transform by default. """
        return Compose([])

    @staticmethod
    def unnormalize(sample):
        return sample


class Bernoulli(ObservationProbabilisticModel):

    def __init__(self, **kw):
        super().__init__(**kw)

    def params_per_dim(self):
        return 1

    def get_mean(self, params):
        dist = D.Bernoulli(logits=params)
        return dist.mean

    def sample_once(self, params, sampling_temperature=1.0):
        dist = D.Bernoulli(logits=params)
        return dist.sample(torch.Size([1])).squeeze(0)

    def get_most_probable_output(self, params):
        dist = D.Bernoulli(logits=params)
        return dist.probs

    def conditional_log_prob(self, params, sample):
        dist = D.Bernoulli(logits=params)
        log_prob = dist.log_prob(sample)
        return log_prob.view(log_prob.size(0), -1).sum(dim=1)

    def get_transform(self):
        """ Compress to [0,1]. """
        return ToTensor()


class DL(ObservationProbabilisticModel):
    """ Discretized Logistic. """

    def sample_once(self, params, sampling_temperature=1.0):
        raise NotImplementedError

    def __init__(self, **kw):
        super().__init__(**kw)
        self.binsize = 1 / (2 ** self.nbits)
        self.logscale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def params_per_dim(self):
        return self.channels

    def get_mean(self, params):
        return params.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

    def conditional_log_prob(self, params, sample):
        # mean and log variance
        mean = self.get_mean(params)
        scale = torch.exp(self.logscale)
        # compute difference of CDFs
        sample = (sample - mean) / scale
        log_prob = torch.log(torch.sigmoid(sample + self.binsize / scale) - torch.sigmoid(sample) + 1e-7)
        return log_prob.view(log_prob.size(0), -1).sum(dim=1)

    def get_transform(self):
        return Compose([ToTensor(), Quantize(self.nbits), Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

    @staticmethod
    def unnormalize(x):
        return x + 0.5


class DML(ObservationProbabilisticModel):
    """ Dicretized Mixture of Logistics. """

    def __init__(self, mix_components=10, **kw):
        super().__init__(**kw)
        self.mix = mix_components
        self.binsize = 1 / (2 ** self.nbits - 1)

    # --- Some auxiliary functions

    def params_per_dim(self):
        return self.mix + self.mix * self.channels * 3

    def unpack_params(self, params):
        """ Unpacking the parameters of DML distribution."""
        # first extract logit probabilities
        # worked in PyTorch 1.6
        # logit_probs, other_params = params.split([self.mix, self.mix * self.channels * 3], -1)
        # to avoid in-place operation in PyTorch 1.7:
        logit_probs = params[..., :self.mix]
        other_params = params[..., self.mix:]
        # extract other parameters
        other_params = other_params.reshape(params.size(0), self.image_size, self.image_size, self.channels, 3*self.mix)
        # worked in PyTorch 1.6
        # means, log_scales, coeffs = other_params.split([self.mix, self.mix, self.mix], -1)
        # to avoid in-place operation in PyTorch 1.7:
        means = other_params[..., :self.mix]
        log_scales = other_params[..., self.mix:2*self.mix]
        coeffs = other_params[..., 2*self.mix:]
        # apply additional transformations, as done in the official repo
        log_scales = torch.clamp(log_scales.clone(), min=-7.)
        coeffs = torch.tanh(coeffs)
        # return
        return logit_probs, means, log_scales, coeffs

    def to_one_hot(self, tensor, fill_with=1.):
        # we perform one hot encore with respect to the mixture (last) axis
        one_hot = torch.zeros(tensor.size() + (self.mix,), device=tensor.device)
        one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
        return one_hot

    @staticmethod
    def log_sum_exp(x):
        """ numerically stable log_sum_exp implementation that prevents overflow """
        m, _ = torch.max(x, dim=-1)
        m2, _ = torch.max(x, dim=-1, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m2), dim=-1))

    @staticmethod
    def log_prob_from_logits(x):
        """ numerically stable log_softmax implementation that prevents overflow """
        m, _ = torch.max(x, dim=-1, keepdim=True)
        return x - m - torch.log(torch.sum(torch.exp(x - m), dim=-1, keepdim=True))

    @staticmethod
    def correlate_subpixels(image, coeffs):
        image[:, :, :, 0] = torch.clamp(image[:, :, :, 0], min=-1, max=1)
        image[:, :, :, 1] = torch.clamp(image[:, :, :, 1] + coeffs[:, :, :, 0] * image[:, :, :, 0], min=-1, max=1)
        image[:, :, :, 2] = torch.clamp(image[:, :, :, 2] + coeffs[:, :, :, 1] * image[:, :, :, 0]
                                                          + coeffs[:, :, :, 2] * image[:, :, :, 1], min=-1, max=1)
        return image

    def get_transform(self):
        return Compose([ToTensor(), Quantize(self.nbits), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def unnormalize(x):
        return x * 0.5 + 0.5

    # --- Main functions

    def conditional_log_prob(self, params, sample):

        # to enable easier code migration, TF convention is used
        params = params.permute(0, 2, 3, 1)
        sample = sample.permute(0, 2, 3, 1)

        # unpack parameters of the mixture distribution and artificially introduce 'mixture' dimension into sample
        logit_probs, means, log_scales, coeffs = self.unpack_params(params)
        sample = sample.unsqueeze(-1) + torch.zeros_like(means)

        # Adjust means by correlating sub-pixels explicitly
        means[:, :, :, 1] = means[:, :, :,  1] + coeffs[:, :, :,  0] * sample[:, :, :,  0]
        means[:, :, :, 2] = means[:, :, :,  2] + coeffs[:, :, :,  1] * sample[:, :, :,  0] \
                                               + coeffs[:, :, :,  2] * sample[:, :, :,  1]

        # Computing the elements of the main equation, the multipliers of the logit probabilities
        centered_x = sample - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + self.binsize)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - self.binsize)
        cdf_min = torch.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # consider edge cases
        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12).clone()) \
                        + (1. - inner_inner_cond) * (log_pdf_mid - np.log(0.5 / self.binsize))
        inner_cond = (sample > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = (sample < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
        log_probs = torch.sum(log_probs, dim=3) + self.log_prob_from_logits(logit_probs)

        log_prob = self.log_sum_exp(log_probs)
        return log_prob.view(log_prob.size(0), -1).sum(dim=1)

    def get_mean(self, params):

        # to enable easier code migration, TF convention is used
        params = params.permute(0, 2, 3, 1)

        # unpack parameters of the mixture distribution and artificially introduce 'mixture' dimension into sample
        logit_probs, means, log_scales, coeffs = self.unpack_params(params)

        # get most probable mixture component, and create a mask
        _, argmax = logit_probs.data.max(dim=-1)
        mix_mask = self.to_one_hot(argmax).unsqueeze(-2)

        # select logistic parameters
        means = torch.sum(means * mix_mask, dim=-1)
        coeffs = torch.sum(coeffs * mix_mask, dim=-1)

        # adjust mean output by correlating sub-pixels
        output = means
        output = self.correlate_subpixels(output, coeffs)

        # permute back
        output = output.permute(0, 3, 1, 2)

        return output

    def sample_once(self, params, sampling_temperature=1.0):

        # to enable easier code migration, TF convention is used
        params = params.permute(0, 2, 3, 1)

        # unpack parameters of the mixture distribution and artificially introduce 'mixture' dimension into sample
        logit_probs, means, log_scales, coeffs = self.unpack_params(params)

        # sample mixture indicator from softmax, and create a mask
        u = torch.zeros_like(logit_probs).uniform_(1e-5, 1. - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(u))
        _, argmax = temp.max(dim=3)
        mix_mask = self.to_one_hot(argmax, self.mix).unsqueeze(-2)

        # select logistic parameters
        means = torch.sum(means * mix_mask, dim=-1)
        log_scales = torch.sum(log_scales * mix_mask, dim=-1)
        coeffs = torch.sum(coeffs * mix_mask, dim=-1)

        # we don't actually round to the nearest 8bit value when sampling
        u = torch.zeros_like(means).uniform_(1e-5, 1. - 1e-5)

        # sample output
        output = means + torch.exp(log_scales) * sampling_temperature * (torch.log(u) - torch.log(1. - u))
        output = self.correlate_subpixels(output, coeffs)

        # permute back
        output = output.permute(0, 3, 1, 2)

        return output