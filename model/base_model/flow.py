from collections.abc import Iterable
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from survae.nn.layers import LambdaLayer
from survae.nn.layers.autoregressive import AutoregressiveShift
from survae.utils import sum_except_batch
from survae.distributions import Distribution, ConditionalNormal, ConditionalDistribution
from survae.transforms import PermuteAxes, Conv1x1, Reverse, Softplus, Bijection
from survae.transforms.surjections import Surjection
from survae.transforms.bijections.conditional import ConditionalBijection
from survae.transforms.bijections.functional import splines


from survae.utils import context_size
from survae.transforms import Transform, ConditionalTransform


eps = 1e-9

class ArgmaxFlow(Distribution):
    def __init__(self, in_dim, hidden_dim, planer_step_num, argmax_step_num=2, num_classes=2):
        super(ArgmaxFlow, self).__init__()
        K = BinaryProductArgmaxSurjection.classes2dims(num_classes)
        context_init_net = IdxContextNet(num_classes=num_classes, hidden_dim=hidden_dim)    # (B,H,L)

        # [B,H,L] --conv--> [B,2*K,L] --split--> mean[B,K,L], std[B,K,L] --sample--> z[B,K,L]
        encoder_base = ConditionalNormal(nn.Conv1d(hidden_dim, 2*K, kernel_size=1, padding=0), split_dim=1)

        # Encoder transforms
        encoder_transforms = []
        for step in range(argmax_step_num):
            if step > 0:
                encoder_transforms.append(Reverse(in_dim, dim=2))
                encoder_transforms.append(Conv1x1(K, slogdet_cpu=False))

            encoder_transforms.append(ConditionalSplineAutoregressive1d(c=K, num_layers=1, hidden_size=hidden_dim,
                                                                        dropout=0.0, num_bins=5,
                                                                        context_size=hidden_dim, unconstrained=True))
        encoder_transforms.append(PermuteAxes([0, 2, 1]))  # (B,K,L) -> (B,L,K)
        encoder = BinaryEncoder(ConditionalInverseFlow(base_dist=encoder_base,
                                                       transforms=encoder_transforms,
                                                       context_init=context_init_net), dims=K)

        self.argmax_transforms = BinaryProductArgmaxSurjection(encoder, num_classes)
        self.planer_transforms = [PlanarTransform(in_dim) for _ in range(planer_step_num)]

    def sample(self, z):  # test
        for planer_transform in self.planer_transforms:
            z, _ = planer_transform(z)
        c = self.argmax_transforms.inverse(z)
        c = c.unsqueeze(-1)
        return c

    def sample_with_log_prob(self, z):    # train
        log_prob = torch.zeros(z.shape[0], device=z.device)
        for planer_transform in self.planer_transforms:
            z, ldj = planer_transform(z)
            log_prob -= ldj
        c = self.argmax_transforms.inverse(z)
        c = c.unsqueeze(-1)

        _, ldj = self.argmax_transforms(c)
        log_prob += ldj
        return c, log_prob



# ===================================================== transform =================================================
class PlanarTransform(Bijection):
    def __init__(self, in_dim, init_sigma=0.01):
        super(PlanarTransform, self).__init__()
        self.u = nn.Parameter(torch.randn(1, in_dim).normal_(0, init_sigma).cuda())
        self.w = nn.Parameter(torch.randn(1, in_dim).normal_(0, init_sigma).cuda())
        self.b = nn.Parameter(torch.randn(1).fill_(0).cuda())

    def forward(self, z):
        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        wtu = (self.w @ self.u.t()).squeeze()
        m_wtu = - 1 + torch.log1p(wtu.exp())
        u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)  # z = x + u h( w^T x + b)

        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b) ** 2) @ self.w
        det = 1 + psi @ u_hat.t()
        ldj = torch.log(torch.abs(det) + 1e-6).squeeze()
        return z, ldj

    def inverse(self, z):
        pass


class BinaryProductArgmaxSurjection(Surjection):
    '''
    A generative argmax surjection using a Cartesian product of binary spaces. Argmax is performed over the final dimension.
    Args:
        encoder: ConditionalDistribution, a distribution q(z|x) with support over z s.t. x=argmax z.
    Example:
        Input tensor x of shape (B, D, L) with discrete values {0,1,...,C-1}:
        encoder should be a distribution of shape (B, D, L, D), where D=ceil(log2(C)).
        When e.g. C=27, we have D=5, such that 2**5=32 classes are represented.
    '''
    stochastic_forward = True

    def __init__(self, encoder, num_classes):
        super(BinaryProductArgmaxSurjection, self).__init__()
        assert isinstance(encoder, ConditionalDistribution)
        self.encoder = encoder
        self.num_classes = num_classes
        self.dims = self.classes2dims(num_classes)

    @staticmethod
    def classes2dims(num_classes):
        return int(np.ceil(np.log2(num_classes)))

    def idx2base(self, idx_tensor):
        return integer_to_base(idx_tensor, base=2, dims=self.dims)

    def base2idx(self, base_tensor):
        return base_to_integer(base_tensor, base=2)

    def forward(self, x):
        z, log_qz = self.encoder.sample_with_log_prob(context=x)
        ldj = -log_qz
        return z, ldj

    def inverse(self, z):
        binary = torch.gt(z, 0.0).long()
        idx = self.base2idx(binary)
        return idx



class InvertSequentialCL():
    '''
    Invert autoregressive bijection in sequential order.
    Data is assumed to be audio / time series of shape (C, L).
    Args:
        shape (Iterable): The data shape, e.g. (2,1024).
        order (str): The order in which to invert. Choices: `{'cl', 'l'}`.
    '''

    def __init__(self, order='cl'):
        assert order in {'cl', 'l'}
        self.order = order
        self.ready = False

    def setup(self, ar_net, element_inverse_fn):
        self.ar_net = ar_net
        self.element_inverse_fn = element_inverse_fn
        self.ready = True

    def inverse(self, z, **kwargs):
        assert self.ready, 'Run scheme.setup(...) before scheme.invert(...).'
        with torch.no_grad():
            if self.order == 'cl': x = self._inverse_cl(z, **kwargs)
            if self.order == 'l': x = self._inverse_l(z, **kwargs)
        return x

    def _inverse_cl(self, z, **kwargs):
        _, C, L = z.shape
        x = torch.zeros_like(z)
        for l in range(L):
            for c in range(C):
                element_params = self.ar_net(x, **kwargs)
                x[:,c,l] = self.element_inverse_fn(z[:,c,l], element_params[:,c,l])
        return x

    def _inverse_l(self, z, **kwargs):
        _, C, L = z.shape
        x = torch.zeros_like(z)
        for l in range(L):
            element_params = self.ar_net(x, **kwargs)
            x[:,:,l] = self.element_inverse_fn(z[:,:,l], element_params[:,:,l])
        return x


class ConditionalAutoregressiveBijection(ConditionalBijection):
    """
    Autoregressive bijection.
    Transforms each input variable with an invertible elementwise bijection,
    conditioned on the previous elements.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.

    Args:
        ar_net: nn.Module, an autoregressive network such that `params = ar_net(x)`.
        scheme: An inversion scheme. E.g. RasterScan from utils.
    """
    def __init__(self, ar_net, scheme):
        super(ConditionalAutoregressiveBijection, self).__init__()
        self.ar_net = ar_net
        self.scheme = scheme
        self.scheme.setup(ar_net=self.ar_net,
                          element_inverse_fn=self._element_inverse)

    def forward(self, x, context):
        params = self.ar_net(x, context=context)
        # print("params:", params.min(), params.max(), params.isnan().sum())
        z, ldj = self._forward(x, params)
        return z, ldj

    def inverse(self, z, context):
        return self.scheme.inverse(z=z, context=context)

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _forward(self, x, params):
        raise NotImplementedError()

    def _element_inverse(self, z, element_params):
        raise NotImplementedError()


class ConditionalLayerLSTM(nn.LSTM):
    def forward(self, x, context):
        output, _ = super(ConditionalLayerLSTM, self).forward(torch.cat([x, context], dim=-1)) # output, (c_n, h_n)
        return output

class ConditionalAutoregressiveLSTM(nn.Module):

    def __init__(self, C, P, num_layers, hidden_size, dropout, context_size):
        super(ConditionalAutoregressiveLSTM, self).__init__()

        self.l_in = LambdaLayer(lambda x: x.permute(2,0,1)) # (B,C,L) -> (L,B,C)
        self.lstm = ConditionalLayerLSTM(C+context_size, hidden_size, num_layers=num_layers, dropout=dropout) # (L,B,C) -> (L,B,H)
        self.l_out = nn.Sequential(nn.Linear(hidden_size, P*C), # (L,B,H) -> (L,B,P*C)
                                   AutoregressiveShift(P*C),
                                   LambdaLayer(lambda x: x.reshape(*x.shape[0:2], C, P)), # (L,B,P*C) -> (L,B,C,P)
                                   LambdaLayer(lambda x: x.permute(1,2,0,3))) # (L,B,C,P) -> (B,C,L,P)


    def forward(self, x, context):
        x = self.l_in(x)
        context = self.l_in(context)

        x = self.lstm(x, context=context)
        return self.l_out(x)


class ConditionalSplineAutoregressive1d(ConditionalAutoregressiveBijection):
    def __init__(self, c, num_layers, hidden_size, dropout, num_bins, context_size, unconstrained):
        self.unconstrained = unconstrained
        self.num_bins = num_bins
        scheme = InvertSequentialCL(order='cl')
        lstm = ConditionalAutoregressiveLSTM(C=c, P=self._num_params(),
                                             num_layers=num_layers,
                                             hidden_size=hidden_size,
                                             dropout=dropout,
                                             context_size=context_size)
        super(ConditionalSplineAutoregressive1d, self).__init__(ar_net=lstm, scheme=scheme)
        self.register_buffer('constant', torch.log(torch.exp(torch.ones(1)) - 1))
        self.autoregressive_net = self.ar_net # For backwards compatability

    def _num_params(self):
        # return 3 * self.num_bins + 1
        return 3 * self.num_bins - 1

    def _forward(self, x, params):
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params[..., 2*self.num_bins:] + self.constant
        if self.unconstrained:
            z, ldj_elementwise = splines.unconstrained_rational_quadratic_spline(
                x,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=False)
        else:
            z, ldj_elementwise = splines.rational_quadratic_spline(
                x,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=False)

        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _element_inverse(self, z, element_params):
        unnormalized_widths = element_params[..., :self.num_bins]
        unnormalized_heights = element_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = element_params[..., 2*self.num_bins:] + self.constant
        if self.unconstrained:
            x, _ = splines.unconstrained_rational_quadratic_spline(
                z,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=True)
        else:
            x, _ = splines.rational_quadratic_spline(
                z,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=True)
        return x


# ===================================================== encoder =================================================
class IdxContextNet(nn.Sequential):
    def __init__(self, num_classes, hidden_dim):
        super(IdxContextNet, self).__init__(
            nn.Embedding(num_classes, hidden_dim),      # [B,L] -> [B,L,H]
            nn.Linear(hidden_dim, hidden_dim),
            LambdaLayer(lambda x: x.permute(0, 2, 1)))  # [B,L,H] -> [B,H,L]

class BinaryEncoder(ConditionalDistribution):
    '''An encoder for BinaryProductArgmaxSurjection.'''

    def __init__(self, noise_dist, dims):
        super(BinaryEncoder, self).__init__()
        self.noise_dist = noise_dist
        self.dims = dims
        self.softplus = Softplus()

    def sample_with_log_prob(self, context):
        # Example: context.shape = (B, I) with values in {0,1,...,K-1}, values表示属于哪个类别
        # Sample z.shape = (B, I, K)

        binary = integer_to_base(context, base=2, dims=self.dims)   # 将context的类别转换为2进制类别表示, (B, C, H, W, K)
        sign = binary * 2 - 1   # 将0/1表示转换为-1/1表示

        u, log_pu = self.noise_dist.sample_with_log_prob(context=context)
        u_positive, ldj = self.softplus(u)

        log_pu_positive = log_pu - ldj
        z = u_positive * sign

        log_pz = log_pu_positive
        return z, log_pz


# ===================================================== utils =================================================
def integer_to_base(idx_tensor, base, dims):
    '''
    Encodes index tensor to a Cartesian product representation.
    Args:
        idx_tensor (LongTensor): An index tensor, shape (...), to be encoded.
        base (int): The base_model to use for encoding.
        dims (int): The number of dimensions to use for encoding.
    Returns:
        LongTensor: The encoded tensor, shape (..., dims).
    '''
    powers = base ** torch.arange(dims - 1, -1, -1, device=idx_tensor.device)
    floored = idx_tensor[..., None] // powers
    remainder = floored % base

    base_tensor = remainder
    return base_tensor


def base_to_integer(base_tensor, base):
    '''
    Decodes Cartesian product representation to an index tensor.
    Args:
        base_tensor (LongTensor): The encoded tensor, shape (..., dims).
        base (int): The base_model used in the encoding.
    Returns:
        LongTensor: The index tensor, shape (...).
    '''
    dims = base_tensor.shape[-1]
    powers = base ** torch.arange(dims - 1, -1, -1, device=base_tensor.device)
    powers = powers[(None,) * (base_tensor.dim()-1)]

    idx_tensor = (base_tensor * powers).sum(-1)
    return idx_tensor



# ================================== FIX: ConditionalNormal ========================================
class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1):
        super(ConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        return Normal(loc=mean, scale=(log_std.exp()+eps))

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev


class ConditionalInverseFlow(ConditionalDistribution):
    """
    Base class for ConditionalFlow.
    Inverse flows use the forward transforms to transform noise to samples.
    These are typically useful as variational distributions.
    Here, we are not interested in the log probability of novel samples.
    However, using .sample_with_log_prob(), samples can be obtained together
    with their log probability.
    """

    def __init__(self, base_dist, transforms, context_init=None):
        super(ConditionalInverseFlow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.context_init = context_init

    def log_prob(self, x, context):
        raise RuntimeError("ConditionalInverseFlow does not support log_prob, see ConditionalFlow instead.")

    def sample(self, context):
        if self.context_init: context = self.context_init(context)
        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(context)
        else:
            z = self.base_dist.sample(context_size(context))
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                z, _ = transform(z, context)
            else:
                z, _ = transform(z)
        return z

    def sample_with_log_prob(self, context):
        if self.context_init: context = self.context_init(context)
        if isinstance(self.base_dist, ConditionalDistribution):
            z, log_prob = self.base_dist.sample_with_log_prob(context)
        else:
            z, log_prob = self.base_dist.sample_with_log_prob(context_size(context))
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                z, ldj = transform(z, context)
            else:
                z, ldj = transform(z)
            log_prob -= ldj
        return z, log_prob