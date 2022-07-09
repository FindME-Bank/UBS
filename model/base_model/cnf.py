from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from model.base_model.ode import ODEfunc, ODEnet

"""
Link: https://github.com/rtqichen/ffjord
Paper: [2019][ICLR] FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models
"""

__all__ = ["build_cnf", "CNF"]


def build_cnf(input_dim: int, hidden_dims: Union[str, list, tuple], layer_type:str, nonlinearity:str,
              strides: Union[str, list, tuple, bool]=None,
              time_length=1.0, regularization_fns=None):
    """
        Args:
            input_dim: ode_net parameter
            hidden_dims: ode_net parameter
            layer_type: ode_net parameter
            nonlinearity: ode_net parameter
            strides: ode_net parameter
            time_length: cnf parameter
            regularization_fns: cnf parameter

        Returns:
            conditional continuous normalizing flow with ode
        """
    hidden_dims = tuple(map(int, hidden_dims.split(","))) if type(hidden_dims)==str else hidden_dims
    strides = tuple(map(int, strides.split(",")))  if type(strides)==str else strides

    diffeq = ODEnet(in_dim=input_dim,  hidden_dims=hidden_dims, out_dim =input_dim, strides=strides,
                    layer_type=layer_type, nonlinearity=nonlinearity)

    odefunc = CNFODEfunc(diffeq=diffeq, divergence_fn="approximate", residual=False, rademacher=True)

    cnf = CNF(odefunc=odefunc, T=time_length, train_T=True, regularization_fns=regularization_fns,
              solver='dopri5', rtol=1e-5, atol=1e-5)
    return cnf


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns={}, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedCNFODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                adjoint_options={"norm": "seminorm"}
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc.num_evals.item()

class CNFODEfunc(ODEfunc):
    def __init__(
            self, diffeq, divergence_fn="approximate", residual=False, rademacher=False
    ):
        super(CNFODEfunc, self).__init__(diffeq=diffeq)
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.0))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(
                np.prod(y.shape[1:]), dtype=torch.float32
            ).to(divergence)
        return tuple(
            [dy, -divergence]
            + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
        )


# class CNFODEfunc(ODEfunc):
#     """
#     包装出一个用于cnf的ode function，用于作为odeint的function输入。主要是完成偏导和trace的工作。
#     Link: https://github.com/rtqichen/ffjord
#     """
#
#     def __init__(self, diffeq: ODEnet, effective_shape, divergence_fn="approximate", residual=False, rademacher=False):
#         """
#         @param diffeq: 一个用于拟合常微分方程的网络
#         @param divergence_fn: 选择trace的计算方法 [brute_force, approximate]
#         @param residual:
#         @param rademacher: 专门用于approximate的trace计算方法
#                            if true，从rademacher分布中sample noise (self._e)
#                            if false，从gaussian分布中sample noise
#         """
#         super(CNFODEfunc, self).__init__(diffeq)
#         assert divergence_fn in ("brute_force", "approximate")
#
#         # self.diffeq = basic_net.wrappers.diffeq_wrapper(diffeq)
#         self.residual = residual
#         self.rademacher = rademacher
#         self._e = None
#
#         # select function for trace calculate
#         if divergence_fn == "brute_force":
#             self.divergence_fn = divergence_bf
#         elif divergence_fn == "approximate":
#             self.divergence_fn = divergence_approx
#
#         self.effective_shape = effective_shape
#
#     def before_odeint(self, e=None):
#         self._e = e
#         self._num_evals.fill_(0)
#
#     def sample_for_hutchinson_estimator(self, y):
#         """
#         Args:
#             y:
#
#         Returns:
#
#         """
#         # Sample and fix the noise.
#         if self._e is None:
#             if self.rademacher:
#                 self._e = sample_rademacher_like(y)
#             else:
#                 self._e = sample_gaussian_like(y)
#
#     def forward(self, t, states):
#         assert len(states) >= 2
#         y = states[0]
#         # increment num evals
#         self._num_evals += 1
#
#         # convert to tensor
#         t = torch.tensor(t).type_as(y)
#         batchsize = y.shape[0]
#
#         # Sample and fix the noise.
#         if self._e is None:
#             self._e = torch.zeros_like(y)
#             if isinstance(self.effective_shape, int):
#                 sample_like = y[:, : self.effective_shape]
#             else:
#                 sample_like = y
#                 for dim, size in enumerate(self.effective_shape):
#                     sample_like = sample_like.narrow(dim + 1, 0, size)
#
#             if self.rademacher:
#                 sample = sample_rademacher_like(sample_like)
#             else:
#                 sample = sample_gaussian_like(sample_like)
#             if isinstance(self.effective_shape, int):
#                 self._e[:, : self.effective_shape] = sample
#             else:
#                 pad_size = []
#                 for idx in self.effective_shape:
#                     pad_size.append(0)
#                     pad_size.append(y.shape[-idx - 1] - self.effective_shape[-idx - 1])
#                 pad_size = tuple(pad_size)
#                 self._e = torch.functional.padding(sample, pad_size, mode="constant")
#             ## pad zeros
#
#         with torch.set_grad_enabled(True):
#             y.requires_grad_(True)
#             t.requires_grad_(True)
#             for s_ in states[2:]:
#                 s_.requires_grad_(True)
#             dy = self.diffeq(t, y, *states[2:])
#             # Hack for 2D data to use brute force divergence computation.
#             if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
#                 divergence = divergence_bf_aug(dy, y, self.effective_shape).view(
#                     batchsize, 1
#                 )
#             else:
#                 divergence = self.divergence_fn(
#                     dy, y, self.effective_shape, e=self._e
#                 ).view(batchsize, 1)
#         if self.residual:
#             dy = dy - y
#             if isinstance(self.effective_dim, int):
#                 divergence -= (
#                         torch.ones_like(divergence)
#                         * torch.tensor(
#                     np.prod(y.shape[1:]) * self.effective_shape / y.shape[1],
#                     dtype=torch.float32,
#                 ).to(divergence)
#                 )
#             else:
#                 divergence -= (
#                         torch.ones_like(divergence)
#                         * torch.tensor(
#                     np.prod(self.effective_shape),
#                     dtype=torch.float32,
#                 ).to(divergence)
#                 )
#         return tuple(
#             [dy, -divergence]
#             + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
#         )


class RegularizedCNFODEfunc(nn.Module):
    """
    wrappers for CNFODEfunc
    # Link: https://github.com/rtqichen/ffjord/lib/layers/wrappers/cnf_regularization.py
    # Paper: [2019][ICLR] FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models
    """
    def __init__(self, odefunc, regularization_fns):
        super(RegularizedCNFODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regularization_fns = regularization_fns

    def before_odeint(self, *args, **kwargs):
        self.odefunc.before_odeint(*args, **kwargs)

    def forward(self, t, state):
        class SharedContext(object):
            pass

        with torch.enable_grad():
            x, logp = state[:2]
            x.requires_grad_(True)
            logp.requires_grad_(True)
            dstate = self.odefunc(t, (x, logp))
            if len(state) > 2:
                dx, dlogp = dstate[:2]
                reg_states = tuple(reg_fn(x, logp, dx, dlogp, SharedContext) for reg_fn in self.regularization_fns)
                return dstate + reg_states
            else:
                return dstate

    @property
    def _num_evals(self):
        return self.odefunc.num_evals


# =================================== func ===================================== #
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]



def divergence_bf(dx, y, effective_dim=None, **unused_kwargs):
    """
    Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    Args:
        dx: Output of the neural ODE function
        y: input to the neural ODE function
        effective_dim: 除去aug dim之后的维度
        **unused_kwargs:

    Returns:
        sum_diag: 求一个trace的可导的解(determin)
    """
    effective_dim = y.shape[1] if effective_dim == None else effective_dim
    assert effective_dim <= y.shape[1]

    sum_diag = 0.
    for i in range(effective_dim):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()



def divergence_approx(f, y, e=None):
    """
        Calculates the approx trace of the Jacobian df/dz.
        link: https://github.com/rtqichen/ffjord/
        Args:
            f: Output of the neural ODE function
            y: input to the neural ode function
            e:
        Returns:
            sum_diag: estimate log determinant of the df/dy 求一个trace的可导的无偏估计解
    """
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def divergence_bf_aug(dx, y, effective_dim, **unused_kwargs):
    """
    The function for computing the exact log determinant of jacobian for augmented ode

    Parameters
        dx: Output of the neural ODE function
        y: input to the neural ode function
        effective_dim (int): the first n dimension of the input being transformed
                             by normalizing flows to compute log determinant
    Returns:
        sum_diag: determin
    """
    sum_diag = 0.0
    assert effective_dim <= y.shape[1]
    for i in range(effective_dim):
        sum_diag += (
            torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0]
                .contiguous()[:, i]
                .contiguous()
        )
    return sum_diag.contiguous()




def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)
