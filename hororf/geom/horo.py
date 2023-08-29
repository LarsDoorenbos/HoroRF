"""Horocycle projection utils (Poincare model)."""

import torch

MIN_NORM = 1e-15


def busemann(x, p, keepdim=True):
    """
    x: (..., d)
    p: (..., d)

    Returns: (..., 1) if keepdim==True else (...)
    """

    xnorm = x.norm(dim=-1, p=2, keepdim=True)
    pnorm = p.norm(dim=-1, p=2, keepdim=True)
    p = p / pnorm.clamp_min(MIN_NORM)

    num = torch.norm(p - x, dim=-1, keepdim=True) ** 2
    den = (1 ** 2 - xnorm ** 2).clamp_min(MIN_NORM)
    ans = torch.log((num / den).clamp_min(MIN_NORM))

    if not keepdim:
        ans = ans.squeeze(-1)
    return ans


def circle_intersection_(r, R):
    """ Computes the intersection of a circle of radius r and R with distance 1 between their centers.

    Returns:
    x - distance from center of first circle
    h - height off the line connecting the two centers of the two intersection pointers
    """

    x = (1.0 - R ** 2 + r ** 2) / 2.0
    s = (r + R + 1) / 2.0
    sq_h = (s * (s - r) * (s - R) * (s - 1)).clamp_min(MIN_NORM)
    h = torch.sqrt(sq_h) * 2.0
    return x, h


def circle_intersection(c1, c2, r1, r2):
    """ Computes the intersections of a circle centered at ci of radius ri.

    c1, c2: (..., d)
    r1, r2: (...)
    """

    d = torch.norm(c1 - c2)  # (...)
    x, h = circle_intersection_(r1 / d.clamp_min(MIN_NORM), r2 / d.clamp_min(MIN_NORM))  # (...)
    x = x.unsqueeze(-1)
    h = h.unsqueeze(-1)
    center = x * c2 + (1 - x) * c1  # (..., d)
    radius = h * d  # (...)

    # The intersection is a hypersphere of one lower dimension, intersected with the plane
    # orthogonal to the direction c1->c2
    # In general, you can compute this with a sort of higher dimensional cross product?
    # For now, only 2 dimensions

    ortho = c2 - c1  # (..., d)
    assert ortho.size(-1) == 2
    direction = torch.stack((-ortho[..., 1], ortho[..., 0]), dim=-1)
    direction = direction / torch.norm(direction, keepdim=True).clamp_min(MIN_NORM)
    return center + radius.unsqueeze(-1) * direction  # , center - radius*direction


def busemann_to_horocycle(p, t):
    """ Find the horocycle corresponding to the level set of the Busemann function to ideal point p with value t.

    p: (..., d)
    t: (...)

    Returns:
    c: (..., d)
    r: (...)
    """
    # Busemann_p(x) = d means dist(0, x) = -d
    q = -torch.tanh(t / 2).unsqueeze(-1) * p
    c = (p + q) / 2.0
    r = torch.norm(p - q, dim=-1) / 2.0
    return c, r


def sphere_intersection(c1, r1, c2, r2):
    """ Computes the intersections of a circle centered at ci of radius ri.

    c1, c2: (..., d)
    r1, r2: (...)

    Returns:
    center, radius such that the intersection of the two spheres is given by
    the intersection of the sphere (c, r) with the hyperplane orthogonal to the direction c1->c2
    """

    d = torch.norm(c1 - c2, dim=-1)  # (...)
    x, h = circle_intersection_(r1 / d.clamp_min(MIN_NORM), r2 / d.clamp_min(MIN_NORM))  # (...)
    x = x.unsqueeze(-1)
    center = x * c2 + (1 - x) * c1  # (..., d)
    radius = h * d  # (...)
    return center, radius


def sphere_intersections(c, r):
    """ Computes the intersection of k spheres in dimension d.

    c: list of centers (..., k, d)
    r: list of radii (..., k)

    Returns:
    center: (..., d)
    radius: (...)
    ortho_directions: (..., d, k-1)
    """

    k = c.size(-2)
    assert k == r.size(-1)

    ortho_directions = []
    center = c[..., 0, :]  # (..., d)
    radius = r[..., 0]  # (...)
    for i in range(1, k):
        center, radius = sphere_intersection(center, radius, c[..., i, :], r[..., i])
        ortho_directions.append(c[..., i, :] - center)
    ortho_directions.append(torch.zeros_like(center))  # trick to handle the case k=1
    ortho_directions = torch.stack(ortho_directions, dim=-1)  # (..., d, k-1) [last element is 0]
    return center, radius, ortho_directions


def project2d(p, q, x):
    # reconstruct p and q in 2D
    p_ = torch.stack([p.new_ones(p.shape[:-1]), p.new_zeros(p.shape[:-1])], dim=-1)
    cos = torch.sum(p * q, dim=-1)
    sin = torch.sqrt(1 - cos ** 2)
    q_ = torch.stack([cos, sin], dim=-1)
    bp = busemann(x, p).squeeze(-1)
    bq = busemann(x, q).squeeze(-1)
    c0, r0 = busemann_to_horocycle(p_, bp)
    c1, r1 = busemann_to_horocycle(q_, bq)
    reconstruction = circle_intersection(c0, c1, r0, r1)
    return reconstruction