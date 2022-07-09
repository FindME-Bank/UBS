from typing import Union, Optional

import torch
import torch.nn as nn

from model.base_model import basic_net
from model.base_model.basic_net import ACTIVATION

__all__ = ["ODEnet", "ODEfunc"]

class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in ode, 用于封装ode_func，有特定的IO
    """

    def __init__(self, in_dim: int, hidden_dims: Union[int, list, tuple], out_dim: int,
                 strides=None, layer_type="concat", nonlinearity="softplus"):
        """
        多层的 diffeq net，base_layer网络分为「线性网络」和「卷积网络」2类
        Args:
            in_dim: 如果是线性网络，表示输入数据的feature维度；
                    如果是卷积网络，表示输入数据的in_channels大小。
            hidden_dims: 如果是int类型，表示是单层网络;
                         如果list/tuple类型，表示是多层网络，其中每一个元素表示每一层的输出维度。
            out_dim: 如果是线性网络，表示输出数据的feature维度；
                     如果是卷积网络，表示输出数据的in_channels大小。
            strides: 如果是线性网络，填写此参数无效；
                     如果是卷积网络，if strides is None，所有卷积层使用basic_net中的默认参数；
                                   if strides is not None，the type of strides must be list，such as [1,2,None],
                                                           其中的每一个元素表示每一层卷积的参数，该元素仅支持这值[1,2,-2,None]。
                                                           当stride==None时，表示使用basic_net中的默认参数；
                                                           当stride==1/2/-2时，具体的卷积的参数见_get_conv_layer_kwarg函数
            layer_type: 单层网络的类型，目前支持的单层网络有 ["ignore", "hyper", "squash", "concat", "concat_v2",
                                                          "concatsquash", "blend", "concatcoord", "conv_ignore",
                                                          "conv_hyper", "conv_squash", "conv_concat", "conv_concat_v2",
                                                          "conv_concatsquash", "conv_blend", "conv_concatcoord"]，
                         其中，如果前缀为"conv"表示卷积网络，反之为线性网络
            nonlinearity: 激活函数的类型。除了最后一层，每一层网络后面会跟着激活函数。
                          目前支持的激活函数有 ["tanh", "softplus", "elu", "swish", "square", "identity"]，
                          如果想要了解具体的激活函数，可以查看lib.func.NONLINEARITIES。
        """
        super(ODEnet, self).__init__()
        self.base_layer = {
            "ignore": basic_net.IgnoreLinear,
            "hyper": basic_net.HyperLinear,
            "squash": basic_net.SquashLinear,
            "concat": basic_net.ConcatLinear,
            "concat_v2": basic_net.ConcatLinear_v2,
            "concatsquash": basic_net.ConcatSquashLinear,
            "blend": basic_net.BlendLinear,
            "concatcoord": basic_net.ConcatLinear,
            "conv_ignore": basic_net.IgnoreConv2d,
            "conv_hyper": basic_net.HyperConv2d,
            "conv_squash": basic_net.SquashConv2d,
            "conv_concat": basic_net.ConcatConv2d,
            "conv_concat_v2": basic_net.ConcatConv2d_v2,
            "conv_concatsquash": basic_net.ConcatSquashConv2d,
            "conv_blend": basic_net.BlendConv2d,
            "conv_concatcoord": basic_net.ConcatCoordConv2d,
        }[layer_type]
        self.layer_type = [None] * (len(hidden_dims) + 1) if strides is None else strides
        self.strides = strides
        self.nonlinearity = nonlinearity

        # dim process
        hidden_dims = [hidden_dims] if type(hidden_dims) is int else hidden_dims
        dims = [in_dim] + hidden_dims + [out_dim]

        # build layers and add them
        self.layers, self.activation_fns = self._bulid_layers(dims)

    def _get_conv_layer_kwargs(self, stride):
        """
        这一丢参数适合cv数据，不适合时序
        """
        if stride is None:
            layer_kwargs = {}
        elif stride == 1:
            layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
        elif stride == 2:
            layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
        elif stride == -2:
            layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
        else:
            raise ValueError('Unsupported stride: {}'.format(stride))
        return layer_kwargs

    def _bulid_layers(self, dims):
        """
        Args:
            dims: 如果是int类型，表示是单层网络;
                  如果list/tuple类型，表示是多层网络，其中每一个元素表示每一层的输入维度。
        Returns:
            layers: 2个nn.ModuleList，分别是layers和activation_fns,
                    它们交替构成：layer1->activation1->layer2->....->activation_dim-2->layer_dim-1
        """
        layers = []
        activation_fns = []

        for i in range(len(dims) - 1):
            if "conv" in self.layer_type:    # conv网络
                layer_kwargs = self._get_conv_layer_kwargs(self.strides[i])
                layers.append(self.base_layer(dims[i], dims[i + 1], **layer_kwargs))
            else:   # linear网络
                layers.append(self.base_layer(dims[i], dims[i + 1]))

            # if i < len(dims) - 2:
            activation_fns.append(ACTIVATION[self.nonlinearity])

        return nn.ModuleList(layers), nn.ModuleList(activation_fns)

    def forward(self, t, x):
        for l, layer in enumerate(self.layers):
            x = layer(t, x)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                x = self.activation_fns[l](x)
        return x


class ODEfunc(nn.Module):
    """
    包装出一个ode function，用于作为odeint的function输入。主要是完成偏导工作。
    """

    def __init__(self, diffeq: Optional[ODEnet]):
        super(ODEfunc, self).__init__()

        self.diffeq = diffeq

        self.register_buffer("_num_evals", torch.tensor(0.))

    def num_evals(self):
        """

        Returns: 调用的次数，用于计算nfe的值

        """
        return self._num_evals.item()

    def forward(self, t, states):
        """

        Args:
            t:
            states: y

        Returns: dy_dt

        """

        return self.diffeq(states)



