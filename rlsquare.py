from numbers import Number
from functools import wraps
import math
from typing import Union

import numpy as np
from scipy.special import factorial
import scipy.stats as ss
import torch
from torch import nn
from torch import distributions as td
import torch.nn.functional as F

TINY = 1e-30
TOL = 1e-8

lgamma = torch.lgamma

def ensure_torch_tensor(value):
    "cast as torch tensor if not already torch tensor"
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    return value


def incompbeta(a, b, x, maxiter=200, tol=1e-8):
    '''
    from https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
    incompbeta(a,b,x) evaluates incomplete beta function, here a, b > 0 and 0 <= x <= 1. This function requires contfractbeta(a,b,x, ITMAX = 200)
    (Code translated from: Numerical Recipes in C.)'''

    a = ensure_torch_tensor(a).float()
    b = ensure_torch_tensor(b).float()
    x = ensure_torch_tensor(x).float()

    lbeta = lgamma(a+b) - lgamma(a) - lgamma(b) + a * torch.log(x) + b * torch.log(1.-x)
    left = torch.exp(lbeta) * contfractbeta(a, b, x, maxiter, tol) / a
    right = 1. - torch.exp(lbeta) * contfractbeta(b, a, 1.-x, maxiter, tol) / b

    # choose the side with fast convergence
    output = torch.where(x < (a + 1.) / (a + b + 2.), left, right)

    # if (x < (a+1.) / (a+b+2.)):
    #     output = torch.exp(lbeta) * contfractbeta(a, b, x, maxiter, tol) / a;
    # else:
    #     output = 1. - torch.exp(lbeta) * contfractbeta(b, a, 1.-x, maxiter, tol) / b;
    # fix 0 and 1 probs
    output = torch.where((x == 0.) + (x == 1.), x, output)
    return output


def contfractbeta(a, b, x, maxiter=200, tol=1e-8):
    """ contfractbeta() evaluates the continued fraction form of the incomplete Beta function; incompbeta().
    (Code translated from: Numerical Recipes in C.)"""

    bm = az = am = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab*x / qap

    for i in range(maxiter + 1):
        em = float(i+1)
        tem = em + em
        d = em*(b-em)*x / ((qam+tem)*(a+tem))
        ap = az + d*am
        bp = bz + d*bm
        d = -(a+em)*(qab+em)*x / ((qap+tem)*(a+tem))
        app = ap + d*az
        bpp = bp + d*bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if (torch.abs(az-aold) < (tol * torch.abs(az))).all():
            return az
    print(f'WARNING: did not all converge to tolerence {tol}')
    return az


class Beta(td.Beta):
    """
    extension of torch.distributions.Beta in following ways:
        + concentrations default init uniform distributed
        + calculates continuous apprx to CDF with default
            params meant for speed and convergence via amortization
            eg like CD1 for contrastive divergence or m=1 for VAEs
    """
    def __init__(self,
                 concentration1=1.,
                 concentration0=1.,
                 maxiter=1, tol=1e-2):
        super().__init__(
            concentration1=ensure_torch_tensor(concentration1),
            concentration0=ensure_torch_tensor(concentration0))
        self.maxiter = maxiter
        self.tol = tol

    def cdf(self, value):
        return incompbeta(a=self.concentration1,
                           b=self.concentration0,
                           x=value,
                           maxiter=self.maxiter,
                           tol=self.tol)


def ensure_dim(value, dim=None, return_dim=False):
    if isinstance(value, Number):
        dim = dim or 1
        value = torch.ones(dim) * value
    if dim is not None:
        assert len(value) == dim
    else:
        dim = len(value)
    if return_dim:
        return value, dim
    else:
        return value


def ensure_torch_tensor_input(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = [ensure_torch_tensor(arg) for arg in args]
        kwargs = {key: ensure_torch_tensor(val) for key, val in kwargs.items()}
        return fn(*args, **kwargs)
    return wrapper


def softplus(x, TINY=1e-8):
    return F.softplus(x) + TINY

e_ = torch.exp(torch.tensor(1.)).item()


class RichardsDiffEq(nn.Module):
    """solution to Richard's diff equation"""
    def __init__(self, dim=None,
                 log_alpha=e_, log_v=e_,
                 Y0=0., t0=0., K=1., TINY=1e-3):
        super().__init__()
        self.log_alpha, dim = ensure_dim(log_alpha, dim, return_dim=True)
        self.log_v = ensure_dim(log_v, dim)
        self.Y0 = ensure_dim(Y0, dim)
        self.t0 = ensure_dim(t0, dim)
        self.K = ensure_dim(K, dim)
        self.TINY = TINY

        self.Y0 = self.Y0.clamp(self.TINY, 1.-self.TINY)

    @property
    def v(self):
        return softplus(self.log_v, self.TINY)

    @property
    def alpha(self):
        return softplus(self.log_alpha, self.TINY)

    @property
    def Q(self):
        return self.K.div(self.Y0).sub(1.).pow(self.v.pow(-1))
        # return -1. + (self.K / self.Y0) ** self.v

    def denom_exp(self, t):
        return -self.alpha * self.v * (t - self.t0)

    def denom(self, t):
        output = (1. + self.Q * softplus(self.denom_exp(t), self.TINY))
        return output.pow(1. / self.v)

    def forward(self, t):
        return self.K / self.denom(t)


class HSVBeta(Beta):
    """
    multivar beta for sampling hsvs
    goodies include:
        + takes beta param logits for easier optimization
        + param vals are softplus(logit) + TINY to meet param
            constraints and for numerical stability
        + auto-inits beta param logits at uniform if not explicitely provided
        + checks for proper sized params for HSV, including mixing new and
          pretrained params
    """
    def __init__(self,
                 dim=None,
                 concentration1_logits=None,
                 concentration0_logits=None,
                 TINY=1e-8):
        if concentration1_logits is None:
            concentration1_logits = torch.randn(1, 3)
        assert concentration1_logits.shape[-1] == 3
        if concentration0_logits is None:
            concentration0_logits = torch.randn(1, 3)
        assert concentration0_logits.shape[-1] == 3

        if dim is not None:
            assert concentration1_logits.shape[-2] == dim
            assert concentration0_logits.shape[-2] == dim
        else:
            assert concentration1_logits.shape[-2] == concentration0_logits.shape[-2]
            dim = concentration1_logits.shape[-2]
        self.dim = dim

        self.hsv_concentration1_logits = nn.Parameter(concentration1_logits)
        self.hsv_concentration0_logits = nn.Parameter(concentration0_logits)

        self.TINY = TINY

        super().__init__(concentration1=self.hsv_concentration1,
                         concentration0=self.hsv_concentration0)

    @property
    def hsv_concentration1(self):
        return softplus(self.hsv_concentration1_logits)

    @property
    def hsv_concentration0(self):
        return softplus(self.hsv_concentration0_logits)

    def softplus(self, x):
        return F.softplus(x) + self.TINY


class BetaCDF(nn.Module):
    """cdf of a beta distribution parameterized on init.
    intended to be used as an expressive class of a monotonic
    functions on [0, 1]"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.beta = Beta(*args, **kwargs)

    def forward(self, steps: Union[torch.LongTensor, torch.FloatTensor],
                return_steps=False)\
            -> torch.FloatTensor:
        if isinstance(steps, torch.LongTensor):
            # assumes each val in steps is the number of steps from 0 to 1
            #   you want to take"
            if len(steps.shape) > 0:
                steps = [torch.linspace(0, 1, steps_) for steps_ in steps]
                weights = torch.cat([self.beta.cdf(steps_) for steps_ in steps])
            else:
                steps = torch.linspace(0, 1, steps)
                weights = self.beta.cdf(steps)
        else:
            # calculates cdf as each val in steps
            weights = self.beta.cdf(steps)
        if return_steps:
            return weights, steps
        else:
            return weights



# transition_color01 = RichardsDiffEq()
# transition_color10 = RichardsDiffEq()

color1 = HSVBeta()
color0 = HSVBeta()

transition_color01 = BetaCDF()
transition_color10 = BetaCDF()

c1 = color1.rsample()
c2 = color0.rsample()

w10 = transition_color10(torch.tensor(100))
w01 = transition_color10(torch.tensor(100))
