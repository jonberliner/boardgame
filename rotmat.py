from functools import lru_cache

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy import ndimage
import geoopt.linalg as gl

from torch.nn import functional as F
from torch import nn
import torch

rng = np.random.RandomState()

pimg = Image.open('/Users/jsb/Downloads/FullSizeRender.jpg')

img = np.array(pimg) / 255.

# from http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
class LeakyTanH(nn.Module):
    def __init__(self, alpha: float=0.01) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        output = self.tanh(inputs)
        return output.where(output < 0., output * self.alpha)


def sparse_sparse_mm(sm1, sm2):
    orsm1 = sm1[sm1.argsort(dim=0)]
    orocsm1 = orsm1




class RotMat(nn.Module):
    def __init__(self, dim_input, dim_output=None):
        dim_output = dim_output or dim_input
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.zeros = torch.zeros(dim_input, dim_output)

    def forward(self, theta: torch.FloatTensor, dims: torch.LongTensor):
        assert len(theta.shape) == 1
        batch_size = len(theta)
        assert dims.shape[0] == batch_size
        assert dims.shape[1] == 2
        assert len(dims.shape) == 2

        # batch_size
        batch_idxs = torch.arange(batch_size)

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # batch_size x 4
        vals = torch.cat([costheta, sintheta, -sintheta, costheta], 1)

        # batch_size x dim x dim
        coos = {}
        for m in range(self.dim-1):
            coos[m] = {}
            for n in range(m, self.dim):
                sm = torch.sparse_coo_tensor(indices=[[m,n], [n, m]],
                                             values=[-1., 1.])
                coos[m][n] = sm
        coos = torch.stack(coos)






    @property
    def Y(self, m, n):
        torch.sparse_coo_tensor(indices=[[m,n], [n, m]],
                                values=[-1., 1.],
                                size=self.dim, self.dim)

        tmp =






        all_.index_copy(dim=1, dims)
        inds = torch.cat([batch_idxs.unsqueeze(1), dims], 1)


        all_ = torch.sparse_coo_tensor(batch_size, self.dim, self.dim)
        all_.scatter(dim=0, dims, vals)

        all_.scatter(dim=1, dims, vals)

        i_xy = torch.tensor([[1, 1],
                             [1, 1],
                             [1, 0],
                             [1, 1]])  #.unsqueeze(0).expand(batch_size, 4, 2)
        dims[i_xy].shape

        dims_ = dims.unsqueeze(1).expand(batch_size, 4, 2)

        i_xy.bmm()

        inds = dims.view(batch_size, 1, 2).expand(-1, 4, -1).mul(i_xy)

        # batch_size x 3


        inds = batch_idxs.view(-1, 1, 1).expand(-1, 2, 2)

        torch.sparse_coo_tensor(inds, 2, 2)


        rotmat = torch.stack([
            torch.stack([costheta, sintheta]),
            torch.stack([-sintheta, costheta])])


        vals = [[torch.cos(theta), torch.sin(theta)],
                [-torch.sin(theta), torch.cos(theta)]]
        torch.zeros()


class QRMatrix(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)


class OrthogonalLinear(nn.Linear):
    """linear restricted to exploring an orthogonal space"""
    def __init__(self, in_features, out_features, bias=True,
                 gain=1., learnable_gain=False):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)

        self._weight_seed = nn.Parameter(
            torch.randn(self.in_features, self.out_features))

        self._gain = nn.Parameter(
            torch.tensor(gain), requires_grad=learnable_gain)

    @property
    def gain(self):
        return F.softplus(self._gain)

    @lru_cache(maxsize=16)
    def weight_seed(self, _weight_seed=None):
        if self.in_features < self.out_features:
            out = out.t()
        return out

    @lru_cache(maxsize=16)
    def qr(self, weight_seed=None):
        weight_seed = weight_seed or self.weight_seed()
        q, r = torch.qr(weight_seed)
        return q, r

    @lru_cache(maxsize=16)
    @staticmethod
    def ph(r, scale=None):
        """tanh soft apprx to sign function of r from qr"""
        scale = scale or torch.tensor(1.)
        d = torch.diag(r, 0)
        return torch.tanh(d * scale)

    @lru_cache(maxsize=1)
    def _weight(self):
        # Compute the qr factorization
        q, r = self.qr()

        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        ph = self.ph(r)
        q = q * ph

        if self.out_features < self.in_features:
            q = q.t()

        W = q.mul(F.softplus(self._gain))
        return W

    @property
    # @lru_cache(maxsize=1)
    def weight(self):
        # Compute the qr factorization
        q, r = self.qr()

        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        ph = self.ph(r)
        q = q * ph

        if self.out_features < self.in_features:
            q = q.t()

        W = q.mul(F.softplus(self._gain))
        return W


def reverse(tensor, dim=0):
    inv_idx = torch.arange(tensor.size(dim)-1, -1, -1).long()
    # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
    inv_tensor = tensor.index_select(dim, inv_idx)
    # or equivalently
    inv_tensor = tensor[inv_idx]
    return inv_tensor


class SVDLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)
        self.U = OrthogonalLinear(in_features, in_features, bias=False)
        self.V = OrthogonalLinear(out_features, out_features, bias=False)

        self.diag_neg_delta = nn.Parameter(
            torch.diag(torch.randn(in_features, out_features)))
        self.dim_diag = min(in_features, out_features)
        self.diag_bias_logit = nn.Parameter(torch.tensor(1.))
        self.min_bias = min_bias

    @property
    def diag(self):
        logits = self.diag_neg_delta
        # negative + pos bias because have to flip to get in descending order
        diag_bias = F.softplus(self.diag_bias_logit)
        diag_delta = -torch.cumsum(F.softplus(self.diag_neg_delta), dim=0)
        diag_ = diag_bias + diag_delta
        return diag_

    @property
    def diagmat(self):
        return self._diagmat(self.diag)

    def _diagmat(self, diag):
        """take a vector of diag values and return diag mat of
        appropriate rectangular size"""
        output = torch.zeros(self.in_features, self.out_features)
        output[range(self.dim_diag), range(self.dim_diag)] =  diag
        return output


    @property
    def weight(self):
        return self.udv_to_weight(
            U=self.U.weight,
            D=self.diagmat,
            V=self.V.weight.t())

    def udv_to_weight(self, U, D, V):
        if len(D.shape) == 1:
            D = self._diagmat(D)
        out = U @ D @ V.t()
        return out.t()



class GLF(nn.Module):
    """from https://en.wikipedia.org/wiki/Generalised_logistic_function"""
    def __init__(self,
                 a=0.,
                 k=1.,
                 b_logit=1.,
                 m=0.5,
                 q=0.5,
                 inv_v_logit=0.):
        self.a = to_param(a, False)  # lower limit
        self.k = to_param(k, False)  # upper limit
        self.c = to_param(1., False)  # scalar for carrying capacity
        self.b_logit = to_param(b_logit, True)  # growth rate
        self.m = to_param(m, True)   # start time
        self.q = to_param(q, True)  # start value
        self.inv_v_logit = to_param(inv_v_logit, True)  # lower means growth starts later

    def forward(self, inputs):
        b = F.softplus(self.b_logit)
        denom = (self.c + self.q * torch.exp(-self.b * (inputs - self.m)))
        denom = denom.pow(F.softplus(self.inv_v_logit))
        numer = self.k - self.a
        return self.a + numer.div(denom)


def to_param(inputs, requires_grad=True):
    inputs = torch.tensor(inputs)
    return nn.Parameter(inputs, requires_grad=requires_grad)




class EigLinear(SVDLinear):
    def __init__(self, num_features, bias=True):
        self.num_features = num_features
        super().__init__(in_features=num_features,
                         out_features=num_features,
                         bias=bias)
    @property
    def V(self):
        return self.U.t()

    @property
    def E(self):
        "eigenvector matrix"
        return self.U

    @property
    def S(self):
        "eigenvalue matrix"
        evals = self.eig(eigenvectors=False)
        return self._diagmat(evals)

    def eig(self, eigenvectors=False):
        "to mirror torch.Tensor.eig"
        evals = self.diag.pow(2.)
        if eigenvectors:
            evecs = self.U.weight
            return evals, evecs
        return evals

    def symeig(self, eigenvectors=False, upper=None):
        return self.eig(eigenvectors=eigenvectors)


class ApprxRank(nn.Module):
    def __init__(self,
                 in_features: int,
                 apprx_features:int,
                 eigval_thresh: torch.tensor(0.)):
                 # min_eigval: Optional[torch.FloatTensor]=torch.tensor(0.),
                 # max_eigval: Optional[torch.FloatTensor]=torch.tensor(1.)):
        self.eigval_thresh = torch.tensor(eigval_thresh)
        # self.eig_linear = EigLinear(
        #     in_features=in_features,
        #     out_features=apprx_features,
        #     bias=False)

        # self.apprx_features = apprx_features
        # self.min_eigval = min_eigval
        # self.max_eigval = max_eigval

    def forward(self, mat, eigval1=None):
        eigval1 = eigval1 or mat.eig()[0]
        mat = mat / eigval1
        mat = torch.sigmoid(mat - eigval_thresh)

        P = self.eig_linear.udv_to_weight(U=E, D=S, V=E.t())

        apprx_rank = P.trace()





    @staticmethod
    def in_interval(mat, a=-1., b=1.):
        return mat.ge(a) * mat.le(b)

    def indicator_in_interval_ab(self, mat: torch.FloatTensor)\
            -> torch.FloatTensor:
        """take matrix A and return float matrix of values in {0., 1.},
        where output[i][j] = A[i][j] in [a, b]"""

        in_interval = self.in_interval(mat, self.a, self.b)
        return torch.where(in_interval,
                           torch.ones_like(A),
                           torch.zeros_like(A))

    @property
    def TINY(self):
        return 1e-6


    def forward(self, A):
        diag = A.diag
        eigval1 = diag[0] ** 2.




        B = A.div(eval1)


        h = self.indicator_in_interval_ab(mat)


def standardize(mat, dim):
    mean = mat.mean(dim=dim)
    norm = mat.norm(dim=dim)
    return (mat - mean) / norm


class LanczosEigenEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mat, n_components=1):


    def K(self, mat, n_components=1):
        v = standardize(torch.rand(mat.shape[0], 1) * 2. - 1., dim=1)

        K = [v]
        for k in range(1, n_components):
            K.append(mat @ K[k-1])
        K = torch.cat(K, 1)
        return K

    def Q(self, mat, n_components=1):
        K = self.K(mat, n_components=n_components)
        return K.qr('Q')

    def T(self, mat, n_components=1):
        Q = self.Q(mat, n_components=n_components)
        return Q.t() @ mat @ Q



class HutchinsonTraceEstimator(nn.Module):
    def __init__(self, n_sample: int=2):
        super().__init__()
        self.n_sample = n_sample

    def forward(self, mat, n_sample=None):
        n_sample = n_sample or self.n_sample
        nrow, ncol = mat.shape
        inputs = torch.rand(n_sample, ncol) * 2. - 1.
        inputs = inputs - inputs.mean(dim=1)
        inputs = inputs.div(inputs.norm(dim=1))

        outputs = inputs.t() @ mat @ mat
        apprx_trace = outputs.sum() / n_sample
        return apprx_trace

class RotMat(nn.Module):
    def __init__(self, dim_input, dim_output=None):
        dim_output = dim_output or dim_input
        self.dim_input = dim_input
        self.dim_output = dim_output


class AntiSymmetricLinear(nn.Linear):
    def __init__(self, dim_input, dim_output,
                 scale=None,
                 Y_tril_logits=None,
                 P_diag_logits=None,
                 learnable_scale=False,
                 learnable_Y=True,
                 learnable_P=True):
        super().__init__(dim_input=dim_input)

        self.P = nn.Parameter(torch.rand(dim_output))

        self.dim = min(dim_input, dim_output)

        offset = dim_output - dim_input

        mgen = torch.ones(self.dim, self.dim) * 2. * np.pi
        ngtm = torch.ones_like(self._lam) * (np.pi / 2.)
        self.lam_scale = mgen.triu() + ngtm.tril(-1)

        self.Y_tril_inds = torch.tril_indices(dim_input, dim_output, offset)
        self.Y_tril_rows = self.Y_tril_inds[0]
        self.Y_tril_cols = self.Y_tril_inds[1]

        if scale is None:
            scale = torch.tensor(1.)

        if Y_tril_logits is None:
            Y_tril_logits = torch.randn(len(self.tril_inds))

        if P_diag_logits is None:
            P_diag_logits = torch.rand(self.dim)

        self.scale = nn.Parameter(
            torch.tensor(1.),
            requires_grad=learnable_scale)

        self.Y_tril_logits = nn.Parameter(
            Y_tril_logits,
            requires_grad=learnable_Y)

        self.P_diag_logits = nn.Parameter(
            P_diag_logits,
            requires_grad=learnable_P)

        self.Y = torch.zeros(dim_input, dim_output)
        self.P = nn.Parameter(torch.rand(dim_input) * 2. - 1.)

        self._lam = nn.Parameter(torch.randn(dim_input, dim_output))

    @property
    def lam(self):

        self._lam

        torch.sigmoid(self._lam)\
                 * torch.ones_like(self._lam)\
                 * 2. * np.pi\
                 *




    def new_Y(self):
        return torch.eye(self.dim_input, self.dim_output)

    def new_P(self):
        return torch.eye(self.dim_output, self.dim_output)

    @property
    def Y(self):
        out = self.new_Y()
        out[]

    # n >= m
    @property
    def n_inner(self, n, m):




    @property
    def unitary_operator(self, n):
        torch.exp(self.scale * self.P[n, n] @ )

    @property
    def weight(self):
        mat = self._weight[self.tril_inds, self.tril_cols] = self.tril
        mat = mat - mat.t()
        return mat



    @property
    def weight(self):


dim = 5

n_mats = (dim * (dim - 1)) // 2

layers = []
for imat in range(n_mats):
    mat =
def

def rot2d(array: np.array, axis1: int, axis2: int):
    return array


def rand_rotation_matrix(deflection, theta_, phi_, z_):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    theta_, phi_, z_: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    num_mats = len(deflection)

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, np.zeros_like(ct),
                  (-st, ct, np.zeros_like(ct)),
                  (, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M



def rotate(rgbs, deg, axes):
    rbg_rot =
    assert len(axes) == 2
    ndimage.rotate(rgbs, deg, )

def add_color_rotate(arr, r_mag, theta_mag):
    _theta = (rng.uniform(-theta_mag, theta_mag) * 2 * np.pi) - np.pi
    theta = _theta * r_mag

    ww, hh, ch = arr.shape
    pixels = arr.reshape(-1, ch)
    pixels = rotate(pixel, theta)

    rotate(arr, theta)

theta =


mx = 1.
mn = -1.

# norm = nimgLj

def add_noise(arr, noise_min, noise_max, arr_min=0., arr_max=1.):
    arr = np.clip(arr, arr_min, arr_max)
    noise = rng.rand(*arr.shape) * (noise_min - noise_max) + (noise_min)
    arr = np.clip(arr + noise, arr_min, arr_max)
    return arr


mag = np.linalg.norm(img, axis=2)
mag /= mag.ravel().max()

noise_max = mag / 2
moise_min = -noise_max


# plt.ion()
fig, ax = plt.subplots()
_img = nimg
canvas = ax.imshow(nimg)

def init():
    canvas.set_data(_img)
    return canvas,

def update_fig(i, _img, ax, canvas):
    _img = add_noise(_img, noise_min=noise_min, noise_max=noise_max)
    canvas.set_data(_img)
    ax.set_title(str(i))
    return canvas,


anim = animation.FuncAnimation(
    fig,
    update_fig,
    frames=1800,
    init_func=init,
    fargs=(_img, ax, canvas),
    blit=True)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='JSB'), bitrate=1800)

anim.save('anim.mp4', writer=writer)


# for step in range(2000):
#     ax.set_title(str(step))
#     _img = add_noise(_img, _mx=_mx, _mn=_mn)
#     canvas.set_data(np.clip(_img, 0., 1.))
#     plt.draw()
#     plt.pause(0.05)
