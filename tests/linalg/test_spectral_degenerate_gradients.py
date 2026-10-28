# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# SVD and eigh gradients at repeated and zero singular values / eigenvalues.
#
# The vector adjoints divide by the gaps s_j^2 - s_i^2 (w_j - w_i for eigh),
# and by S itself for the complex phase term and the projection terms of a
# non-square SVD. A repeated or zero value made that 1/0, and the 0 * inf
# that followed turned the whole gradient into NaN with no warning, even for
# a loss that does not depend on the basis of the repeated subspace -- every
# gradient of an MPS simulation, which starts from a product state. Such a
# loss has a well-defined gradient, and it is what these tests check against.
#
# Ground truth: loss = ||U_2 S_2 Vh_2||_F^2 = s_1^2 + s_2^2, whose gradient
# is 2 * A_2 (A_2 the rank-2 truncation), so long as s_2 > s_3. For eigh,
# loss = tr(P A) with P the projector onto the top-2 eigenspace has gradient
# P^T (folded onto the lower triangle for complex input); a degenerate pair
# and a split pair spanning the same subspace give the same gradient.
import numpy as np
import pytest
import jittor as jt
from jittor import linalg

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if (
        request.param == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


def _unitary(rng, n, complex_):
    z = rng.normal(size=(n, n))
    if complex_:
        z = z + 1j * rng.normal(size=(n, n))
    return np.linalg.qr(z)[0]


def _top2_svd_loss_grad(a_np, dtype):
    a = jt.array(a_np.astype(dtype))
    u, s, vh = linalg.svd(a)
    loss = (jt.matmul(u[:, :2] * s[:2].unsqueeze(-2), vh[:2, :])).abs().sqr().sum()
    return jt.grad(loss, a)


_CASES = {
    "distinct": [4.0, 3.0, 2.0, 1.0],
    "repeated": [2.0, 2.0, 1.0, 0.5],
    "zero": [3.0, 2.0, 0.0, 0.0],
}
_ATOL = {"float32": 1e-4, "complex64": 1e-4, "complex128": 1e-11}


# (m, n) = (4, 6) and (6, 4) add the (I - U U^H) gU S^-1 and
# S^-1 gVh (I - V V^H) terms, which divide by the zero singular values too.
@pytest.mark.parametrize("shape", [(4, 4), (6, 4), (4, 6)])
@pytest.mark.parametrize("dtype", ["float32", "complex64", "complex128"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_svd_gradient_at_repeated_and_zero_singular_values(device, case, dtype, shape):
    m, n = shape
    rng = np.random.default_rng(1)
    complex_ = dtype.startswith("complex")
    u = _unitary(rng, m, complex_)[:, :4]
    v = _unitary(rng, n, complex_)[:, :4]
    s = _CASES[case]
    g = _top2_svd_loss_grad(u @ np.diag(s) @ v.conj().T, dtype).numpy()
    expected = 2 * u[:, :2] @ np.diag(s[:2]) @ v[:, :2].conj().T
    assert np.isfinite(g).all(), g
    np.testing.assert_allclose(g, expected, atol=_ATOL[dtype])


def test_svd_second_derivative_at_repeated_singular_values_is_finite(device):
    rng = np.random.default_rng(3)
    u = _unitary(rng, 4, True)
    a = jt.array((u @ np.diag([2.0, 2.0, 1.0, 0.0]) @ u.conj().T).astype("complex64"))
    u_, s, vh = linalg.svd(a)
    loss = (jt.matmul(u_[:, :2] * s[:2].unsqueeze(-2), vh[:2, :])).abs().sqr().sum()
    g = jt.grad(loss, a)
    gg = jt.grad(g.abs().sqr().sum(), a).numpy()
    assert np.isfinite(gg).all(), gg


def _eigh_projector_loss_grad(h, dtype):
    a = jt.array(h.astype(dtype))
    _, v = linalg.eigh(a)
    p = jt.matmul(v[:, 2:], v[:, 2:].transpose(-1, -2).conj())
    loss = (p * a.transpose(-1, -2)).sum()
    if dtype.startswith("complex"):
        loss = loss.real
    return jt.grad(loss, a).numpy()


@pytest.mark.parametrize("dtype", ["float32", "complex64", "complex128"])
def test_eigh_gradient_at_repeated_eigenvalues_matches_a_split_pair(device, dtype):
    rng = np.random.default_rng(2)
    u = _unitary(rng, 4, dtype.startswith("complex"))
    repeated = _eigh_projector_loss_grad(u @ np.diag([0.5, 1.0, 2.0, 2.0]) @ u.conj().T, dtype)
    split = _eigh_projector_loss_grad(u @ np.diag([0.5, 1.0, 1.99, 2.01]) @ u.conj().T, dtype)
    assert np.isfinite(repeated).all(), repeated
    np.testing.assert_allclose(repeated, split, atol=_ATOL[dtype])
