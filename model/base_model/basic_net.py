from functools import reduce
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

__all__ = ["ACTIVATION",
           # ======================== linear based  ========================
           "MLP", "HyperLinear", "IgnoreLinear", "ConcatLinear", "ConcatLinear_v2",
           "SquashLinear", "ConcatSquashLinear",  "GatedLinear", "BlendLinear",
           # ========================== cnn based  =========================
           "CNN", "HyperConv2d", "IgnoreConv2d", "SquashConv2d", "ConcatConv2d",
           "ConcatConv2d_v2", "ConcatSquashConv2d", "ConcatCoordConv2d", "GatedConv",
           "GatedConvTranspose", "GatedConv2d", "GatedConvTranspose2d",
           # ========================== gnn based  =========================
           "GraphNetBase", "GCN", "GAT"]

ACTIVATION = {
    "elu": nn.ELU(inplace=True),
    "relu": nn.ReLU(inplace=True),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus()
}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


# =========================== linear based ================================  #
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, activation=None, last_activation=None,
                 is_bn: bool = False, dropout: float = 0):
        super(MLP, self).__init__()
        net_list = []
        dims = [in_dim] + hidden_dims
        self.freeze_bn = True

        for i in range(len(hidden_dims)):
            net_list.append(nn.Linear(dims[i], dims[i + 1]))
            if is_bn:
                net_list.append(nn.BatchNorm1d(dims[i + 1]))
            if i < len(hidden_dims) - 1:
                if activation is not None:
                    net_list.append(ACTIVATION[activation])
                if dropout != 0:
                    net_list.append(nn.Dropout(dropout))
            else:
                if last_activation is not None:
                    net_list.append(ACTIVATION[last_activation])

        self.net = nn.Sequential(*net_list)

    def forward(self, x):
        return self.net(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(MLP, self).train(mode)
        # if self.freeze_bn:
        #     print("Freezing Mean/Var of BatchNorm1D.")
        # if self.freeze_bn_affine:
        #     print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    # if self.freeze_bn_affine:
                    #     m.weight.requires_grad = False
                    #     m.bias.requires_grad = False

class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t

# ============================== cnn based ================================ #
class CNN(nn.Module):
    def __init__(self, in_shapes: Union[list, tuple], hidden_channels: list, mlp_hidden_dims: list,
                 kernel_sizes: list = None, strides=None, last_activation=None):
        super(CNN, self).__init__()
        layer_num = len(hidden_channels)
        in_channels = [in_shapes[0]] + hidden_channels[:-1]

        if kernel_sizes is None:
            kernel_sizes = [3] * len(hidden_channels)
        if strides is None:
            strides = [1] * len(hidden_channels)
        multi_strides = reduce(lambda x, y: x * y, strides)

        self.multi_conv = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(in_channels[i], hidden_channels[i],
                      kernel_sizes[i], strides[i], (kernel_sizes[i] - 1) // 2),
            nn.BatchNorm2d(hidden_channels[i]),
            nn.ReLU(inplace=True)) for i in range(layer_num)])

        self.linear = MLP(
            in_dim=hidden_channels[-1] * (in_shapes[1] // multi_strides) * (in_shapes[2] // multi_strides),
            hidden_dims=mlp_hidden_dims, last_activation=last_activation)

    def forward(self, x):
        h = self.multi_conv(x)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return h


class HyperConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(HyperConv2d, self).__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, "dim_in and dim_out must both be divisible by groups."
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d

        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation
        )


class IgnoreConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(IgnoreConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        return self._layer(x)


class SquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(SquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(1, -1, 1, 1)


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatConv2d_v2(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatSquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(1, -1, 1, 1) \
            + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatCoordConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatCoordConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 3, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        b, c, h, w = x.shape
        hh = torch.arange(h).to(x).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).to(x).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.to(x).view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )
        self.layer_g = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendConv2d(nn.Module):
    def __init__(
        self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(BlendConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._layer1 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class GatedConv2d(nn.Module):
    """
    vae basic net
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))
        return h * g


class GatedConvTranspose2d(nn.Module):
    """
    vae basic net
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding,
                 output_padding=0, dilation=1, activation=None):
        super(GatedConvTranspose2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size, stride, padding, output_padding, dilation=dilation
        )
        self.g = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size, stride, padding, output_padding, dilation=dilation
        )

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))
        return h * g


# ============================== classical_gnn based =============================== #
class GraphNetBase(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, activation: str=None, last_activation=None, dropout=0.0, **kwargs):
        super(GraphNetBase, self).__init__()
        self.dropout = dropout
        dims = [in_dim] + hidden_dims
        self.graph_net_list = self.generate_net(dims, **kwargs)
        self.activation_list = nn.ModuleList([ACTIVATION[activation] for _ in range(len(hidden_dims) - 1)]) if activation is not None else None
        self.last_activation = ACTIVATION[last_activation] if last_activation is not None else None

    def generate_net(self, dims, **kwargs):
        raise NotImplemented

    def forward(self, x, edge_index):
        for i in range(len(self.graph_net_list)):
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.graph_net_list[i](x, edge_index)
            if i < len(self.graph_net_list) - 1 and self.activation_list is not None:
                x = self.activation_list[i](x)
            else:
                if self.last_activation is not None:
                    x = self.last_activation(x)
        return x


class GCN(GraphNetBase):
    def __init__(self, in_dim: int, hidden_dims: list, activation: str, last_activation=None):
        super(GCN, self).__init__(in_dim=in_dim, hidden_dims=hidden_dims, activation=activation,
                                  last_activation=last_activation)

    def generate_net(self, dims, **kwargs):
        return nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])


class GAT(GraphNetBase):
    def __init__(self, in_dim: int, hidden_dims: list, activation: str=None, heads: list=None, dropout=0.0,
                 last_activation=None, last_concat=False):
        """

        Returns:
            object:
        """
        self.last_concat = last_concat
        super(GAT, self).__init__(in_dim=in_dim, hidden_dims=hidden_dims, heads=heads, activation=activation,
                                  last_activation=last_activation, dropout=dropout)
        assert len(hidden_dims) == len(heads)

    def generate_net(self, dims, heads):
        if heads is None:
            heads = [1] * len(dims)
        else:
            heads = [1] + heads

        layers = []
        for i in range(len(dims) - 1):
            if i==len(dims) - 2:    # last layer
                layers.append(GATConv(dims[i] * heads[i], dims[i + 1], heads[i+1], concat=self.last_concat,
                                      dropout=self.dropout))
            else:
                layers.append(GATConv(dims[i] * heads[i], dims[i + 1], heads[i+1], dropout=self.dropout))
        return nn.ModuleList(layers)


















