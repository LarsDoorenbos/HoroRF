# https://github.com/leymir/hyperbolic-image-embeddings/blob/master/hyptorch/pmath.py#L441
import torch


def p2k(x, c):
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom


def k2p(x, c):
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denom


def lorenz_factor(x, *, c=1.0, dim=-1, keepdim=False):
    """

    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate Lorenz factor
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorenz factor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def poincare_mean(x, dim=0, c=1.0):
    x = p2k(x, c)
    lamb = lorenz_factor(x, c=c, keepdim=True)
    mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(
        lamb, dim=dim, keepdim=True
    )
    mean = k2p(mean, c)
    return mean.squeeze(dim)