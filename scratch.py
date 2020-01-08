from numbers import Number
from functools import wraps
import math

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


def incompbeta(a, b, x, maxiter=200, tol=1e-8):
    '''
    from https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
    incompbeta(a,b,x) evaluates incomplete beta function, here a, b > 0 and 0 <= x <= 1. This function requires contfractbeta(a,b,x, ITMAX = 200)
    (Code translated from: Numerical Recipes in C.)'''

    a = ensure_torch_tensor(a).float()
    b = ensure_torch_tensor(b).float()
    x = ensure_torch_tensor(x).float()
    if x.eq(0.):
        return x.clone();
    elif x.eq(1.):
        return x.clone();
    else:
        lbeta = lgamma(a+b) - lgamma(a) - lgamma(b) + a * torch.log(x) + b * torch.log(1.-x)
        if (x < (a+1.) / (a+b+2.)):
            return torch.exp(lbeta) * contfractbeta(a, b, x, maxiter, tol) / a;
        else:
            return 1. - torch.exp(lbeta) * contfractbeta(b, a, 1.-x, maxiter, tol) / b;


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
        if (torch.abs(az-aold) < (tol * torch.abs(az))):
            return az
    raise ValueError(f'did not converge to tolerence {tol}')


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


def ensure_torch_tensor(value):
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    return value


def ensure_torch_tensor_input(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = [ensure_torch_tensor(arg) for arg in args]
        kwargs = {key: ensure_torch_tensor(val) for key, val in kwargs.items()}
        return fn(*args, **kwargs)
    return wrapper


# @ensure_torch_tensor_input
# def lbeta(a, b):
#     a = a.float()
#     b = b.float()
#     return lgamma(a) + lgamma(b) - lgamma(a + b)


# def beta(a, b):
#     return torch.exp(lbeta(a, b))


# def incbeta_terms(a, b, x, m):
#     for ii in range(m):
#         # even term
#         if m == 0:
#             even_term = torch.tensor(1.)
#         else:
#             numer = (m * (b - m) * x)
#             denom = (a + 2.*m - 1.) * (a + 2.*m)
#             even_term = numer / denom

#         # odd term
#         numer = (a + m) * (a + b + m) * x
#         denom = (a + 2.*m) * (a + 2.*m + 1.)
#         odd_term = -(numer / denom)




# def incomplete_beta(a, b, x,
#                     maxiter=100,
#                     tol=1e-10
#                     ):
#     x = x.float()
#     a = a.float()
#     b = b.float()

#     assert x >= 0. and x <= 1.
#     if x.eq(1.):
#         return x.clone()

#     # ensure fast convergence via symmetry
#     if x > (a - 1.) / (a + b - 2.):
#         return 1. - incomplete_beta(b, a, 1-x, maxiter, tol)

#     lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b)
#     front = torch.exp((torch.log(x) * a)\
#                       + (torch.log(1. - x) * b)\
#                       - lbeta_ab)\
#             / a

#     # lets apprx w Letnz' algorithm
#     f, c, d = [torch.tensor(v) for v in [1., 1., 0.]]

#     for i in range(maxiter):
#         m = 0.5 * i

#         if i == 0:
#             numer = torch.tensor(1.)
#         elif i % 2 == 0:  # even terms
#             numer =\
#                 (m * (b - m) * x)\
#                 / ((a + 2.*m - 1.) * (a + 2.*m))
#         else:  # odd terms
#             numer =\
#                 -((a + m) * (a + b + m) * x)\
#                 / ((a + 2.*m) * (a + 2.* m+ 1.))
#         print(f'{m}: {numer.item(), c.item(), d.item()}')

#         # lentz alg iteration
#         ddenom = 1. + numer * d
#         if abs(ddenom) < TINY:
#             ddenom = TINY
#         d = 1. / ddenom

#         c = 1. + (numer / c)
#         if abs(c) < TINY:
#             c = TINY
#         # d = 1. + numer * d
#         # d = d if abs(d) > TINY else TINY
#         # d = 1. / d
#         # c = 1. + numer / c
#         # c = c if abs(c) > TINY else TINY
#         cd = c * d
#         f = f * cd
#         if abs(1. - cd) < tol:
#             return front * (f - 1.)
#     print(f'WARNING!: did not converge to tol {tol}')
#     return front * (f - 1.)

# @ensure_torch_tensor_input
# def incbeta(x, a, b, maxiter=100, tol=1e-8):
#     return incomplete_beta(x, a, b, maxiter, tol)


#TINY = 1e-8




#a = (2. * pi * n)
#A = a ** (.5)
#b = (n / e)
#B = b ** (n)

#log(a**0.5) = 0.5 * log(a)

#b = n / e
#log(b) = log(n) - log(e)

#log(b ** n) = n * log(b)


#tau = np.pi * 2.

#def logfac_sterling(n: torch.Tensor) -> torch.Tensor:
#    "log of the sterling apprx to n!"

#    log_left = 0.5 * torch.log(tau * n)
#    log_right = n * torch.log(n / e_)

#    return log_left + log_right


#def fac_sterling(n):
#    "sterling apprx to n! as show here: https://en.wikipedia.org/wiki/Stirling%27s_approximation"
#    n = torch.tensor(n)
#    left = torch.tensor(2.).mul(np.pi).mul(n).sqrt()
#    right = (n / e_).pow(n)
#    return left * right


#def loggamma_sterling(n):
#    "gamma fn apprx using sterling apprx for factorial"
#    return logfac_sterling(n - 1)

#def logbeta_sterling(a, b):
#    return loggamma_sterling(a) + loggamma_sterling(b) - loggamma_sterling(a + b)

#def falling_factorial(n, k):
#    ks = torch.arange(0, n-1)
#    vals = n - ks
#    return vals.prod()


#def pochhammer(q, n):
#    assert n >= 0
#    hi = q * (q + 1)
#    lo = q + n - 1
#    vals = torch.arange(lo, hi)
#    vals = torch.where(vals == 0, torch.ones_like(vals), vals)
#    return vals.sum()



#def softplus(x, TINY=1e-8):
#    return F.softplus(x) + TINY


#class Mish(nn.Module):
#    """
#    Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#    https://arxiv.org/abs/1908.08681v1
#    implemented for PyTorch / FastAI by lessw2020
#    github: https://github.com/lessw2020/mish
#    """
#    def __init__(self):
#        super().__init__()

#    def forward(self, x):
#        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
#        return x *( torch.tanh(F.softplus(x)))


#class RichardCurve(nn.Module):
#    """generalized logistic function"""
#    def to_tensor(self, x):
#        if isinstance(x, Number):
#            x = torch.ones(self.dim) * x
#        x = torch.tensor(x)
#        assert isinstance(x, torch.Tensor)
#        assert len(x) == self.dim
#        return x

#    def __init__(self, dim=None,
#                 B=1., M=1., v=1., Q=1.,
#                 A=0., K=1., C=1., TINY=1e-8):
#        super().__init__()
#        self.dim = dim or 1

#        # TYPICALLY LEARNABLE
#        # growth rate
#        self.B = nn.Parameter(self.to_tensor(B), requires_grad=True)
#        # starting time
#        self.M = nn.Parameter(self.to_tensor(M), requires_grad=True)
#        # max-growth time parameter
#        self.v_ = nn.Parameter(self.to_tensor(v), requires_grad=True)
#        # Y(t=0) parameter
#        self.Q = nn.Parameter(self.to_tensor(Q), requires_grad=True)

#        # TYPICALLY STATIC
#        # lower limit
#        self.A = nn.Parameter(self.to_tensor(A), requires_grad=False)
#        # upper limit
#        self.K = nn.Parameter(self.to_tensor(K), requires_grad=False)
#        # typically 1 or upper-asymptote gets less direct
#        self.C = nn.Parameter(self.to_tensor(C), requires_grad=False)
#        # for stability
#        self.TINY = nn.Parameter(self.to_tensor(TINY), requires_grad=False)

#    @property
#    def numer(self):
#        return self.K - self.A

#    @property
#    def BM(self):
#        return self.B * self.M

#    @property
#    def denom_pow(self):
#        return 1. / self.v

#    @property
#    def v(self):
#        return softplus(self.v_, TINY=TINY)

#    def denom_exp(self, t):
#        return torch.exp(-self.B * t + self.BM)

#    def forward(self, t):
#        denom = (self.C + self.Q * self.denom_exp(t)) ** self.denom_pow
#        return self.A + (self.numer / (denom + self.TINY))


#e_ = torch.exp(torch.tensor(1.)).item()

#class RichardsDiffEq(nn.Module):
#    """solution to Richard's diff equation"""
#    def __init__(self, dim=None,
#                 log_alpha=e_, log_v=e_,
#                 Y0=0., t0=0., K=1., TINY=1e-3):
#        super().__init__()
#        self.log_alpha, dim = ensure_dim(log_alpha, dim, return_dim=True)
#        self.log_v = ensure_dim(log_v, dim)
#        self.Y0 = ensure_dim(Y0, dim)
#        self.t0 = ensure_dim(t0, dim)
#        self.K = ensure_dim(K, dim)
#        self.TINY = TINY

#        self.Y0 = self.Y0.clamp(self.TINY, 1.-self.TINY)

#    @property
#    def v(self):
#        return softplus(self.log_v, self.TINY)

#    @property
#    def alpha(self):
#        return softplus(self.log_alpha, self.TINY)

#    @property
#    def Q(self):
#        return self.K.div(self.Y0).sub(1.).pow(self.v.pow(-1))
#        # return -1. + (self.K / self.Y0) ** self.v

#    def denom_exp(self, t):
#        return -self.alpha * self.v * (t - self.t0)

#    def denom(self, t):
#        output = (1. + self.Q * softplus(self.denom_exp(t), self.TINY))
#        return output.pow(1. / self.v)

#    def forward(self, t):
#        return self.K / self.denom(t)


#class HSVBeta(td.Beta):
#    def __init__(self,
#                 dim=None,
#                 concentration1_logits=None,
#                 concentration0_logits=None,
#                 TINY=1e-8):
#        if concentration1_logits is None:
#            concentration1_logits = torch.randn(1, 3)
#        assert concentration1_logits.shape[-1] == 3
#        if concentration0_logits is None:
#            concentration0_logits = torch.randn(1, 3)
#        assert concentration0_logits.shape[-1] == 3

#        if dim is not None:
#            assert concentration1_logits.shape[-2] == dim
#            assert concentration0_logits.shape[-2] == dim
#        else:
#            assert concentration1_logits.shape[-2] == concentration0_logits.shape[-2]
#            dim = concentration1_logits.shape[-2]
#        self.dim = dim

#        self.hsv_concentration1_logits = nn.Parameter(concentration1_logits)
#        self.hsv_concentration0_logits = nn.Parameter(concentration0_logits)

#        self.TINY = TINY

#        super().__init__(concentration1=self.hsv_concentration1,
#                         concentration0=self.hsv_concentration0)

#    @property
#    def hsv_concentration1(self):
#        return softplus(self.hsv_concentration1_logits)

#    @property
#    def hsv_concentration0(self):
#        return softplus(self.hsv_concentration0_logits)

#    def softplus(self, x):
#        return F.softplus(x) + self.TINY

#logistic10_logits = RichardCurve()
#logistic01_logits = RichardCurve()

#transition_color01 = RichardsDiffEq()
#transition_color10 = RichardsDiffEq()

#color1 = HSVBeta()
#color0 = HSVBeta()
